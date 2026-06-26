import numpy as np
from pyFM.functional import FunctionalMapping
import os
from pathlib import Path
from tqdm.auto import tqdm
import vtk
from PynamicMesh.utils.tools import mesh_mat2object

try:
    import cupy as xp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    GPU_AVAILABLE = False


try:
    from geometrickernels.spaces import Mesh as GKMesh
    from geometrickernels.kernels import MaternKarhunenLoeveKernel
    import lab as B
    GK_AVAILABLE = True
except ImportError:
    GK_AVAILABLE = False


class CustomFunctionalMapping(FunctionalMapping):
    """
    Functional map wrapper with:
      - Combined descriptor support (WKS, HKS, MKS)
      - Matérn Kernel Signatures (MKS) via GeometricKernels
    """

    def _set_descriptors(self, descr1: np.ndarray, descr2: np.ndarray) -> None:
        self.descr1 = descr1
        self.descr2 = descr2
        self.A = descr1
        self.B = descr2

    def _compute_mks(self, mesh_obj, n_descr=100, nu=1.5, min_l=0.01, max_l=10.0, **kwargs):
        """
        Computes the Matérn Kernel Signature (MKS) across multiple lengthscales.
        """
        lengthscales = np.geomspace(min_l, max_l, n_descr)
        
        if GK_AVAILABLE:
            try:
                gk_mesh = GKMesh(mesh_obj.vertices, mesh_obj.faces)
                kernel = MaternKarhunenLoeveKernel(gk_mesh, num_eigenfunctions=len(mesh_obj.eigenvalues))
                

                if hasattr(gk_mesh, '_eigenvalues'):
                    gk_mesh._eigenvalues = mesh_obj.eigenvalues
                    gk_mesh._eigenfunctions = mesh_obj.eigenvectors

                X = np.arange(len(mesh_obj.vertices)).reshape(-1, 1)
                mks_descriptors = []


                if hasattr(kernel, "init_params_and_state"):
                    params, state = kernel.init_params_and_state()
                    for l in lengthscales:
                        params["nu"] = np.array([nu])
                        params["lengthscale"] = np.array([l])
                        diag = kernel.K_diag(params, state, X)
                        mks_descriptors.append(B.to_numpy(diag).flatten())
                else:
                    for l in lengthscales:
                        params = {"nu": np.array([nu]), "lengthscale": np.array([l])}
                        diag = kernel.K_diag(params, X)
                        mks_descriptors.append(np.array(diag).flatten())
                        
                return np.column_stack(mks_descriptors)
            except Exception as e:
                pass

        power = -(nu + 1.0)
        phi_sq = mesh_obj.eigenvectors ** 2
        evals = mesh_obj.eigenvalues
        
        mks_descriptors = []
        for l in lengthscales:
            weights = ((2 * nu) / (l**2) + evals) ** power
            sig = np.sum(phi_sq * weights, axis=1)
            mks_descriptors.append(sig)
            
        return np.column_stack(mks_descriptors)

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
          - "WKS", "HKS", "MKS"
          - Permutations combined with '+': "WKS+HKS", "WKS+MKS", "WKS+HKS+MKS"
        """

        required_k1 = max(K[0], k_process if k_process else 100)
        required_k2 = max(K[1], k_process if k_process else 100)

        if self.mesh1.eigenvalues is None or len(self.mesh1.eigenvalues) < required_k1:
            self.mesh1.process(k=required_k1)
        if self.mesh2.eigenvalues is None or len(self.mesh2.eigenvalues) < required_k2:
            self.mesh2.process(k=required_k2)


        requested_types = [t.strip().upper() for t in str(descr_type).split('+')]
        is_combined = len(requested_types) > 1

        def normalize_descriptor(desc):
            desc_gpu = xp.asarray(desc) if GPU_AVAILABLE else desc
            mean = desc_gpu.mean(axis=0)
            std = desc_gpu.std(axis=0) + 1e-8
            norm_gpu = (desc_gpu - mean) / std
            return norm_gpu.get() if GPU_AVAILABLE else norm_gpu

        d1_blocks = []
        d2_blocks = []

        for dtype in requested_types:
            if dtype == "MKS":
                nu = kwargs.get("nu", 1.5)
                min_l = kwargs.get("min_l", 0.01)
                max_l = kwargs.get("max_l", 10.0)
                d1 = self._compute_mks(self.mesh1, n_descr=n_descr, nu=nu, min_l=min_l, max_l=max_l, **kwargs)
                d2 = self._compute_mks(self.mesh2, n_descr=n_descr, nu=nu, min_l=min_l, max_l=max_l, **kwargs)
            elif dtype in ["WKS", "HKS"]:
                super().preprocess(
                    K=K,
                    n_descr=n_descr,
                    descr_type=dtype,
                    landmarks=landmarks,
                    subsample_step=subsample_step,
                    k_process=required_k1,
                    verbose=verbose,
                    **kwargs,
                )
                d1 = self.descr1.copy()
                d2 = self.descr2.copy()
            else:
                raise ValueError(f"Unsupported descriptor segment type: {dtype}")

            if is_combined:
                d1 = normalize_descriptor(d1)
                d2 = normalize_descriptor(d2)

            d1_blocks.append(d1)
            d2_blocks.append(d2)

        self._set_descriptors(
            np.hstack(d1_blocks),
            np.hstack(d2_blocks),
        )

        return self