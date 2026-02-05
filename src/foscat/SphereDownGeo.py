"""
SphereDownGeo.py - Optimized HEALPix downsampling with Parallel Processing and Caching

OPTIMIZATION STRATEGY:
- Replaces the serial loop over coarse pixels with a PARALLEL loop using joblib.
- Uses the EXACT same `hp.query_disc(inclusive=True)` logic as the original to guarantee
  numerically identical results (unlike KD-tree approximations).
- Adds disk caching.

Expected speedup: Linear with core count (e.g. ~50x on 64 cores).
"""

import os
import hashlib
import pickle
from pathlib import Path
import time

import torch
import torch.nn as nn
import numpy as np
import healpy as hp

# Parallel processing
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Cache directory for downgrade matrices
_DOWNGRADE_CACHE_DIR = Path(os.environ.get("FOSCAT_CACHE", Path.home() / ".FOSCAT" / "cache" / "downgrade"))


class SphereDownGeo(nn.Module):
    """
    Geometric HEALPix downsampling operator (NESTED indexing).

    This module reduces resolution by a factor 2:
        nside_out = nside_in // 2

    Optimizations (this version):
    - Parallel execution of hp.query_disc via joblib (fast + exact).
    - Disk caching.
    """

    def __init__(
        self,
        nside_in: int,
        mode: str = "smooth",
        radius_deg: float | None = None,
        sigma_deg: float | None = None,
        weight_norm: str = "l1",
        cell_ids_out: np.ndarray | list[int] | None = None,
        in_cell_ids: np.ndarray | list[int] | torch.Tensor | None = None,
        use_csr=True,
        device=None,
        dtype: torch.dtype = torch.float32,
        # New parameters for optimization
        use_cache: bool = True,
        verbose: bool = True,
        n_jobs: int = -1,  # Number of parallel jobs (-1 = all cores)
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.dtype = dtype
        self.use_cache = use_cache
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.nside_in = int(nside_in)
        assert (self.nside_in & (self.nside_in - 1)) == 0, "nside_in must be a power of 2."
        self.nside_out = self.nside_in // 2
        assert self.nside_out >= 1, "nside_out must be >= 1."

        self.N_in = 12 * self.nside_in * self.nside_in
        self.N_out = 12 * self.nside_out * self.nside_out

        self.mode = str(mode).lower()
        assert self.mode in ("smooth", "maxpool"), "mode must be 'smooth' or 'maxpool'."

        self.weight_norm = str(weight_norm).lower()
        assert self.weight_norm in ("l1", "l2"), "weight_norm must be 'l1' or 'l2'."

        # ---- Handle reduced-domain inputs (fine pixels) ----
        self.in_cell_ids = self._validate_in_cell_ids(in_cell_ids)
        self.has_in_subset = self.in_cell_ids is not None
        if self.has_in_subset:
            # derive parents
            derived_out = np.unique(self.in_cell_ids // 4).astype(np.int64)
            if cell_ids_out is None:
                self.cell_ids_out = derived_out
            else:
                req_out = self._validate_cell_ids_out(cell_ids_out)
                # keep only those compatible with derived_out (otherwise they'd be all-zero)
                self.cell_ids_out = np.intersect1d(req_out, derived_out, assume_unique=False)
                if self.cell_ids_out.size == 0:
                    raise ValueError(
                        "After intersecting cell_ids_out with unique(in_cell_ids//4), "
                        "no coarse pixel remains. Check your inputs."
                    )
        else:
            self.cell_ids_out = self._validate_cell_ids_out(cell_ids_out)

        self.K_out = int(self.cell_ids_out.size)

        # Column basis for smooth matrix:
        # - full sphere: columns are 0..N_in-1
        # - subset: columns are 0..K_in-1 aligned to self.in_cell_ids
        self.K_in = int(self.in_cell_ids.size) if self.has_in_subset else self.N_in

        if self.mode == "smooth":
            if radius_deg is None:
                # default: include roughly the 4 children footprint
                radius_deg = 2.0 * hp.nside2resol(self.nside_out, arcmin=True) / 60.0
            if sigma_deg is None:
                sigma_deg = max(radius_deg / 2.0, 1e-6)

            self.radius_deg = float(radius_deg)
            self.sigma_deg = float(sigma_deg)
            self.radius_rad = self.radius_deg * np.pi / 180.0
            self.sigma_rad = self.sigma_deg * np.pi / 180.0
            
            # Try to load from cache first, otherwise build with fast method
            M = self._load_cached_matrix()
            if M is None:
                if self.verbose:
                    print(f"Building downgrade matrix: {self.K_out} coarse pixels, {self.K_in} fine pixels...")
                # We use the optimized parallel build
                M = self._build_down_matrix_parallel()
                self._save_matrix_to_cache(M)
              
            self.M = M.coalesce()
            
            if use_csr:
                self.M = self.M.to_sparse_csr().to(self.device)

            self.M_size = M.size()

        else:
            # Precompute children indices for maxpool
            children = np.stack(
                [4 * self.cell_ids_out + i for i in range(4)],
                axis=1,
            ).astype(np.int64)  # [K_out, 4] in fine pixel ids (full indexing)

            if self.has_in_subset:
                # map each child pixel id to position in in_cell_ids (compact index)
                pos = self._positions_in_sorted(self.in_cell_ids, children.reshape(-1))
                children_compact = pos.reshape(self.K_out, 4).astype(np.int64)  # -1 if missing
                self.register_buffer(
                    "children_compact",
                    torch.tensor(children_compact, dtype=torch.long, device=self.device),
                )
            else:
                self.register_buffer(
                    "children_full",
                    torch.tensor(children, dtype=torch.long, device=self.device),
                )

        # expose ids as torch buffers for convenience
        self.register_buffer(
            "cell_ids_out_t",
            torch.tensor(self.cell_ids_out.astype(np.int64), dtype=torch.long, device=self.device),
        )
        if self.has_in_subset:
            self.register_buffer(
                "in_cell_ids_t",
                torch.tensor(self.in_cell_ids.astype(np.int64), dtype=torch.long, device=self.device),
            )

    # ================ CACHING METHODS ================
    
    def _get_cache_key(self) -> str:
        """Generate unique cache key based on configuration."""
        config = {
            'nside_in': self.nside_in,
            'nside_out': self.nside_out,
            'radius_deg': getattr(self, 'radius_deg', None),
            'sigma_deg': getattr(self, 'sigma_deg', None),
            'weight_norm': self.weight_norm,
            'K_out': self.K_out,
            'K_in': self.K_in,
            'has_subset': self.has_in_subset,
        }
        # Hashing the ID arrays is important to distinguish different patches
        if self.has_in_subset:
            config['in_hash'] = hashlib.md5(self.in_cell_ids.tobytes()).hexdigest()[:16]
            config['out_hash'] = hashlib.md5(self.cell_ids_out.tobytes()).hexdigest()[:16]
        else:
            # If full sphere, cell_ids_out might still be a subset or full
            config['out_hash'] = hashlib.md5(self.cell_ids_out.tobytes()).hexdigest()[:16]
        
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()

    def _get_cache_path(self) -> Path:
        """Get path to cached matrix."""
        _DOWNGRADE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return _DOWNGRADE_CACHE_DIR / f"downgrade_{self._get_cache_key()}.pkl"

    def _load_cached_matrix(self):
        """Try to load matrix from cache."""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path()
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                rows_t = torch.tensor(data['rows'], dtype=torch.long, device=self.device)
                cols_t = torch.tensor(data['cols'], dtype=torch.long, device=self.device)
                vals_t = torch.tensor(data['vals'], dtype=self.dtype, device=self.device)
                indices = torch.stack([rows_t, cols_t], dim=0)
                if self.verbose:
                    print(f"Loaded cached downgrade matrix from {cache_path}")
                return torch.sparse_coo_tensor(
                    indices, vals_t, size=(self.K_out, self.K_in),
                    device=self.device, dtype=self.dtype
                )
            except Exception as e:
                if self.verbose:
                    print(f"Cache load failed: {e}, rebuilding...")
        return None

    def _save_matrix_to_cache(self, M):
        """Save matrix to cache."""
        if not self.use_cache:
            return
            
        cache_path = self._get_cache_path()
        try:
            M_coo = M.coalesce()
            indices = M_coo.indices().cpu().numpy()
            vals = M_coo.values().cpu().numpy()
            data = {'rows': indices[0], 'cols': indices[1], 'vals': vals}
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            if self.verbose:
                print(f"Saved downgrade matrix to {cache_path}")
        except Exception as e:
            if self.verbose:
                print(f"Cache save failed: {e}")

    # ---------------- validation helpers ----------------
    def _validate_cell_ids_out(self, cell_ids_out):
        """Return a 1D np.int64 array of coarse cell ids (nside_out)."""
        if cell_ids_out is None:
            return np.arange(self.N_out, dtype=np.int64)

        arr = np.asarray(cell_ids_out, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            raise ValueError("cell_ids_out is empty: provide at least one coarse pixel id.")
        arr = np.unique(arr)
        if arr.min() < 0 or arr.max() >= self.N_out:
            raise ValueError(f"cell_ids_out must be in [0, {self.N_out-1}] for nside_out={self.nside_out}.")
        return arr

    def _validate_in_cell_ids(self, in_cell_ids):
        """Return a 1D np.int64 array of fine cell ids (nside_in) or None."""
        if in_cell_ids is None:
            return None
        if torch.is_tensor(in_cell_ids):
            arr = in_cell_ids.detach().cpu().numpy()
        else:
            arr = np.asarray(in_cell_ids)
        arr = np.asarray(arr, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            raise ValueError("in_cell_ids is empty: provide at least one fine pixel id or None.")
        arr = np.unique(arr)
        if arr.min() < 0 or arr.max() >= self.N_in:
            raise ValueError(f"in_cell_ids must be in [0, {self.N_in-1}] for nside_in={self.nside_in}.")
        return arr

    @staticmethod
    def _positions_in_sorted(sorted_ids: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
        """
        For each query_id, return its index in sorted_ids if present, else -1.
        sorted_ids must be sorted ascending unique.
        """
        q = np.asarray(query_ids, dtype=np.int64)
        idx = np.searchsorted(sorted_ids, q)
        idx = np.clip(idx, 0, sorted_ids.size - 1)
        ok = (idx < sorted_ids.size) & (sorted_ids[idx] == q)
        out = np.full(q.shape, -1, dtype=np.int64)
        out[ok] = idx[ok]
        return out

    # ---------------- weights and matrix build ----------------
    
    def _build_down_matrix_parallel(self) -> torch.Tensor:
        """
        Build downsampling matrix using Parallelized hp.query_disc calls.
        
        This uses joblib to distribute the EXACT logic from the original implementation
        across multiple cores. This ensures numerical equivalence while providing significant speedup.
        """
        t_start = time.time()
        
        # Prepare arguments for static method
        nside_in = self.nside_in
        nside_out = self.nside_out
        radius_rad = self.radius_rad
        sigma_rad = self.sigma_rad
        subset_cols = self.has_in_subset
        in_ids = self.in_cell_ids
        cell_ids_out = self.cell_ids_out
        weight_norm = self.weight_norm
        
        # Determine number of jobs
        n_jobs = self.n_jobs if self.n_jobs != 0 else -1
        if n_jobs < 0:
            n_jobs = os.cpu_count() or 1
        
        # Split work into chunks
        # Chunk size is a tradeoff: too small = overhead, too large = load imbalance
        chunk_size = max(64, self.K_out // (n_jobs * 8))
        chunks = []
        for i in range(0, self.K_out, chunk_size):
            end = min(i + chunk_size, self.K_out)
            chunks.append( (i, cell_ids_out[i:end]) )
            
        if self.verbose:
            print(f"  Parallel build: {len(chunks)} chunks on {n_jobs} workers")

        # Define the worker function (must be static/picklable or local)
        # We capture context variables.
        def process_chunk(start_idx, chunk_p_out):
            local_rows = []
            local_cols = []
            local_vals = []
            
            # Re-import inside worker to ensure visibility in all contexts
            import healpy as hp
            import numpy as np

            # Helper for weight normalization (local implementation to avoid self dependence)
            def normalize(w):
                w = np.maximum(w, 0.0)
                if weight_norm == "l1":
                    s = w.sum()
                    return w / s if s > 0 else np.ones_like(w) / max(w.size, 1)
                else: # l2
                    s2 = (w*w).sum()
                    return w / np.sqrt(s2) if s2 > 0 else np.ones_like(w) / max(np.sqrt(w.size), 1.0)

            # Pre-calculate common things if possible, but hp.pix2ang is very fast
            for i, p_out in enumerate(chunk_p_out):
                row_idx = start_idx + i
                
                # 1. Get coarse pixel center
                theta0, phi0 = hp.pix2ang(nside_out, int(p_out), nest=True)
                vec0 = hp.ang2vec(theta0, phi0)

                # 2. Query disc (EXACT same as reference)
                neigh = hp.query_disc(nside_in, vec0, radius_rad, inclusive=True, nest=True)
                neigh = np.asarray(neigh, dtype=np.int64)

                # 3. Filter subset
                if subset_cols:
                    neigh_sorted = np.sort(neigh)
                    neigh = np.intersect1d(neigh_sorted, in_ids, assume_unique=False)

                # 4. Fallback for empty neighbors
                if neigh.size == 0:
                    children = (4 * int(p_out) + np.arange(4, dtype=np.int64))
                    if subset_cols:
                        # manual searchsorted since we can't use self._positions... efficiently here
                        idx = np.searchsorted(in_ids, children)
                        idx = np.clip(idx, 0, in_ids.size - 1)
                        ok = (idx < in_ids.size) & (in_ids[idx] == children)
                        if np.any(ok):
                            neigh = children[ok]
                        else:
                            continue
                    else:
                        neigh = children

                # 5. Compute weights
                theta, phi = hp.pix2ang(nside_in, neigh, nest=True)
                vec = hp.ang2vec(theta, phi)

                dots = np.clip(np.dot(vec, vec0), -1.0, 1.0)
                ang = np.arccos(dots)
                w = np.exp(- 2*(ang / sigma_rad) ** 2)
                w = normalize(w)

                # 6. Store
                if subset_cols:
                    # Map global ID back to compact index
                    idx = np.searchsorted(in_ids, neigh)
                    # We know they exist because of the intersection above
                    for c_compact, val in zip(idx, w):
                        local_rows.append(row_idx)
                        local_cols.append(c_compact)
                        local_vals.append(val)
                else:
                    for c_global, val in zip(neigh, w):
                        local_rows.append(row_idx)
                        local_cols.append(c_global)
                        local_vals.append(val)
            
            return local_rows, local_cols, local_vals

        # Execute parallel
        if HAS_JOBLIB and n_jobs != 1:
            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(process_chunk)(idx, p_out_chunk) for idx, p_out_chunk in chunks
            )
        else:
            # Serial fallback
            results = [process_chunk(idx, p_out_chunk) for idx, p_out_chunk in chunks]

        # Flatten results
        all_rows = []
        all_cols = []
        all_vals = []
        for r, c, v in results:
            all_rows.extend(r)
            all_cols.extend(c)
            all_vals.extend(v)
            
        if self.verbose:
            print(f"  Matrix build completed in {time.time() - t_start:.2f}s")
            
        # Build tensor
        if len(all_rows) == 0:
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            vals_t = torch.zeros((0,), dtype=self.dtype, device=self.device)
            return torch.sparse_coo_tensor(
                indices, vals_t, size=(self.K_out, self.K_in),
                device=self.device, dtype=self.dtype
            ).coalesce()

        rows_t = torch.tensor(all_rows, dtype=torch.long, device=self.device)
        cols_t = torch.tensor(all_cols, dtype=torch.long, device=self.device)
        vals_t = torch.tensor(all_vals, dtype=self.dtype, device=self.device)

        indices = torch.stack([rows_t, cols_t], dim=0)
        M = torch.sparse_coo_tensor(
            indices, vals_t, size=(self.K_out, self.K_in),
            device=self.device, dtype=self.dtype
        ).coalesce()
        
        return M

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            If has_in_subset:
                - [B,C,K_in] (compact, aligned with in_cell_ids) OR [B,C,N_in] (full sphere)
            Else:
                - [B,C,N_in] (full sphere)

        Returns
        -------
        y : torch.Tensor
            [B,C,K_out]
        cell_ids_out : torch.Tensor
            [K_out] coarse pixel ids (nside_out), aligned with y last dimension.
        """
        if x.dim() != 3:
            raise ValueError("x must be [B, C, N]")

        B, C, N = x.shape
        if self.has_in_subset:
            if N not in (self.K_in, self.N_in):
                raise ValueError(
                    f"x last dim must be K_in={self.K_in} (compact) or N_in={self.N_in} (full), got {N}"
                )
        else:
            if N != self.N_in:
                raise ValueError(f"x last dim must be N_in={self.N_in}, got {N}")

        if self.mode == "smooth":

            # If x is full-sphere but M is subset-based, gather compact inputs
            if self.has_in_subset and N == self.N_in:
                x_use = x.index_select(dim=2, index=self.in_cell_ids_t.to(x.device))
            else:
                x_use = x

            # sparse mm expects 2D: (K_out, K_in) @ (K_in, B*C)
            x2 = x_use.reshape(B * C, -1).transpose(0, 1).contiguous()
            y2 = torch.sparse.mm(self.M, x2)
            y = y2.transpose(0, 1).reshape(B, C, self.K_out).contiguous()
            return y, self.cell_ids_out_t.to(x.device)

        # maxpool
        if self.has_in_subset and N == self.N_in:
            x_use = x.index_select(dim=2, index=self.in_cell_ids_t.to(x.device))
        else:
            x_use = x

        if self.has_in_subset:
            # children_compact: [K_out, 4] indices in 0..K_in-1 or -1
            ch = self.children_compact.to(x.device)  # [K_out,4]
            # gather with masking
            # We build y by iterating 4 children with max
            y = None
            for j in range(4):
                idx = ch[:, j]  # [K_out]
                mask = idx >= 0
                # start with very negative so missing children don't win
                tmp = torch.full((B, C, self.K_out), -torch.inf, device=x.device, dtype=x.dtype)
                if mask.any():
                    tmp[:, :, mask] = x_use.index_select(dim=2, index=idx[mask]).reshape(B, C, -1)
                y = tmp if y is None else torch.maximum(y, tmp)
            # If a parent had no valid children at all, it is -inf -> set to 0
            y = torch.where(torch.isfinite(y), y, torch.zeros_like(y))
            return y, self.cell_ids_out_t.to(x.device)

        else:
            ch = self.children_full.to(x.device)  # [K_out,4] full indices
            # gather children and max
            xch = x_use.index_select(dim=2, index=ch.reshape(-1)).reshape(B, C, self.K_out, 4)
            y = xch.max(dim=3).values
            return y, self.cell_ids_out_t.to(x.device)


def clear_downgrade_cache():
    """Clear all cached downgrade matrices."""
    if _DOWNGRADE_CACHE_DIR.exists():
        import shutil
        shutil.rmtree(_DOWNGRADE_CACHE_DIR)
        print(f"Cleared downgrade cache at {_DOWNGRADE_CACHE_DIR}")
