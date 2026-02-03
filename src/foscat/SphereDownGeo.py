"""
SphereDownGeo.py - Optimized HEALPix downsampling with KD-tree and caching

This is a drop-in replacement for foscat's SphereDownGeo.py with:
1. KD-tree spatial queries (replaces slow hp.query_disc loop)
2. Parallel weight computation via joblib
3. Disk caching for instant reloading
4. Progress reporting

Expected speedup: 50-200x (1+ hour -> 1-5 minutes for first run, instant thereafter)
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

# Spatial query
from scipy.spatial import cKDTree

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

    Input conventions
    -----------------
    - If in_cell_ids is None:
        x is expected to be full-sphere: [B, C, N_in]
        output is [B, C, K_out] with K_out = len(cell_ids_out) (or N_out if None).
    - If in_cell_ids is provided (fine pixels at nside_in, NESTED):
        x can be either:
          * compact: [B, C, K_in] where K_in = len(in_cell_ids), aligned with in_cell_ids order
          * full-sphere: [B, C, N_in] (also supported)
        output is [B, C, K_out] where cell_ids_out is derived as unique(in_cell_ids // 4),
        unless you explicitly pass cell_ids_out (then it will be intersected with the derived set).

    Modes
    -----
    - mode="smooth": linear downsampling y = M @ x  (M sparse)
    - mode="maxpool": non-linear max over available children (fast)
    
    Optimizations (this version)
    ----------------------------
    - KD-tree spatial queries instead of per-pixel hp.query_disc()
    - Parallel weight computation using joblib
    - Disk caching to avoid recomputation
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
                # (healpy pixel size ~ sqrt(4pi/N), coarse pixel is 4x area)
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
                M = self._build_down_matrix_fast()
                self._save_matrix_to_cache(M)
              
            self.M = M.coalesce()
            
            if use_csr:
                self.M = self.M.to_sparse_csr().to(self.device)

            self.M_size = M.size()

        else:
            # Precompute children indices for maxpool
            # For subset mode, store mapping from each parent to indices in compact vector,
            # with -1 for missing children.
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
        if self.has_in_subset:
            config['in_hash'] = hashlib.md5(self.in_cell_ids.tobytes()).hexdigest()[:16]
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
        ok = (idx >= 0) & (idx < sorted_ids.size) & (sorted_ids[idx] == q)
        out = np.full(q.shape, -1, dtype=np.int64)
        out[ok] = idx[ok]
        return out

    # ---------------- weights and matrix build ----------------
    def _normalize_weights(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float64)
        if w.size == 0:
            return w
        w = np.maximum(w, 0.0)

        if self.weight_norm == "l1":
            s = w.sum()
            if s <= 0.0:
                return np.ones_like(w) / max(w.size, 1)
            return w / s

        # l2
        s2 = (w * w).sum()
        if s2 <= 0.0:
            return np.ones_like(w) / max(np.sqrt(w.size), 1.0)
        return w / np.sqrt(s2)

    def _build_down_matrix_fast(self) -> torch.Tensor:
        """
        Build downsampling matrix using KD-tree for fast spatial queries.
        
        This is MUCH faster than the original hp.query_disc() loop approach.
        """
        t_start = time.time()
        
        nside_in = self.nside_in
        nside_out = self.nside_out
        sigma_rad = self.sigma_rad
        
        subset_cols = self.has_in_subset
        in_ids = self.in_cell_ids if subset_cols else np.arange(self.N_in, dtype=np.int64)
        
        # Step 1: Get all fine pixel 3D coordinates (VECTORIZED)
        if self.verbose:
            print("  Step 1/4: Computing fine pixel coordinates...")
        t1 = time.time()
        
        fine_vecs = np.array(hp.pix2vec(nside_in, in_ids, nest=True)).T  # Shape: (K_in, 3)
        
        if self.verbose:
            print(f"    Done in {time.time() - t1:.2f}s")
        
        # Step 2: Get all coarse pixel 3D coordinates (VECTORIZED)
        if self.verbose:
            print("  Step 2/4: Computing coarse pixel coordinates...")
        t2 = time.time()
        
        coarse_vecs = np.array(hp.pix2vec(nside_out, self.cell_ids_out, nest=True)).T  # Shape: (K_out, 3)
        
        if self.verbose:
            print(f"    Done in {time.time() - t2:.2f}s")
        
        # Step 3: Build KD-tree from fine pixel coordinates
        if self.verbose:
            print("  Step 3/4: Building KD-tree...")
        t3 = time.time()
        
        tree = cKDTree(fine_vecs)
        
        if self.verbose:
            print(f"    Done in {time.time() - t3:.2f}s")
        
        # Step 4: Query neighbors and compute weights
        if self.verbose:
            print("  Step 4/4: Finding neighbors and computing weights...")
        t4 = time.time()
        
        # Convert angular radius to Euclidean distance on unit sphere
        # For angle θ, the chord length is 2*sin(θ/2)
        euclidean_radius = 2.0 * np.sin(self.radius_rad / 2.0)
        
        # Batch query: find all fine pixels within radius of each coarse pixel
        neighbor_lists = tree.query_ball_point(coarse_vecs, r=euclidean_radius, workers=self.n_jobs)
        
        if self.verbose:
            print(f"    KD-tree query done in {time.time() - t4:.2f}s")
        
        # Compute weights
        t5 = time.time()
        
        if HAS_JOBLIB and self.n_jobs != 1 and self.K_out > 1000:
            # Parallel weight computation
            rows, cols, vals = self._compute_weights_parallel(
                coarse_vecs, fine_vecs, neighbor_lists, sigma_rad, in_ids
            )
        else:
            # Sequential weight computation
            rows, cols, vals = self._compute_weights_sequential(
                coarse_vecs, fine_vecs, neighbor_lists, sigma_rad, in_ids
            )
        
        if self.verbose:
            print(f"    Weight computation done in {time.time() - t5:.2f}s")
        
        # Build sparse tensor
        if len(rows) == 0:
            indices = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            vals_t = torch.zeros((0,), dtype=self.dtype, device=self.device)
            return torch.sparse_coo_tensor(
                indices, vals_t, size=(self.K_out, self.K_in),
                device=self.device, dtype=self.dtype
            ).coalesce()

        rows_t = torch.tensor(rows, dtype=torch.long, device=self.device)
        cols_t = torch.tensor(cols, dtype=torch.long, device=self.device)
        vals_t = torch.tensor(vals, dtype=self.dtype, device=self.device)

        indices = torch.stack([rows_t, cols_t], dim=0)
        M = torch.sparse_coo_tensor(
            indices, vals_t, size=(self.K_out, self.K_in),
            device=self.device, dtype=self.dtype
        ).coalesce()
        
        if self.verbose:
            print(f"  Total matrix build time: {time.time() - t_start:.2f}s")
        
        return M

    def _compute_weights_sequential(self, coarse_vecs, fine_vecs, neighbor_lists, sigma_rad, in_ids):
        """Compute Gaussian weights sequentially."""
        rows = []
        cols = []
        vals = []
        
        report_interval = max(1, self.K_out // 10)
        subset_cols = self.has_in_subset
        
        for r, neighbors in enumerate(neighbor_lists):
            if self.verbose and r % report_interval == 0:
                print(f"      Processing coarse pixel {r}/{self.K_out} ({100*r/self.K_out:.0f}%)")
            
            if len(neighbors) == 0:
                # Fallback to direct children
                p_out = self.cell_ids_out[r]
                children = (4 * int(p_out) + np.arange(4, dtype=np.int64))
                if subset_cols:
                    pos = self._positions_in_sorted(in_ids, children)
                    ok = pos >= 0
                    if not np.any(ok):
                        continue
                    neighbors = pos[ok].tolist()
                else:
                    neighbors = children.tolist()
            
            neighbors = np.array(neighbors, dtype=np.int64)
            
            # Compute angular distances using dot product
            vec0 = coarse_vecs[r]
            vecs = fine_vecs[neighbors]
            dots = np.clip(np.dot(vecs, vec0), -1.0, 1.0)
            ang = np.arccos(dots)
            
            # Gaussian weights (same formula as original)
            w = np.exp(-2.0 * (ang / sigma_rad) ** 2)
            
            # Normalize
            w = self._normalize_weights(w)
            
            # Append to lists
            rows.extend([r] * len(neighbors))
            cols.extend(neighbors.tolist())
            vals.extend(w.tolist())
        
        return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(vals, dtype=np.float64)

    def _compute_weights_parallel(self, coarse_vecs, fine_vecs, neighbor_lists, sigma_rad, in_ids):
        """Compute Gaussian weights in parallel using joblib."""
        
        subset_cols = self.has_in_subset
        cell_ids_out = self.cell_ids_out
        weight_norm = self.weight_norm
        
        def process_chunk(chunk_indices):
            local_rows = []
            local_cols = []
            local_vals = []
            
            for r in chunk_indices:
                neighbors = neighbor_lists[r]
                
                if len(neighbors) == 0:
                    # Fallback to direct children
                    p_out = cell_ids_out[r]
                    children = (4 * int(p_out) + np.arange(4, dtype=np.int64))
                    if subset_cols:
                        pos = SphereDownGeo._positions_in_sorted(in_ids, children)
                        ok = pos >= 0
                        if not np.any(ok):
                            continue
                        neighbors = pos[ok].tolist()
                    else:
                        neighbors = children.tolist()
                
                neighbors = np.array(neighbors, dtype=np.int64)
                
                # Compute angular distances
                vec0 = coarse_vecs[r]
                vecs = fine_vecs[neighbors]
                dots = np.clip(np.dot(vecs, vec0), -1.0, 1.0)
                ang = np.arccos(dots)
                
                # Gaussian weights
                w = np.exp(-2.0 * (ang / sigma_rad) ** 2)
                
                # Normalize (inline to avoid method call overhead)
                w = np.maximum(w, 0.0)
                if weight_norm == "l1":
                    s = w.sum()
                    if s > 0:
                        w = w / s
                    else:
                        w = np.ones_like(w) / max(len(w), 1)
                else:  # l2
                    s2 = (w * w).sum()
                    if s2 > 0:
                        w = w / np.sqrt(s2)
                    else:
                        w = np.ones_like(w) / max(np.sqrt(len(w)), 1.0)
                
                local_rows.extend([r] * len(neighbors))
                local_cols.extend(neighbors.tolist())
                local_vals.extend(w.tolist())
            
            return local_rows, local_cols, local_vals
        
        # Split into chunks
        n_jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        chunk_size = max(100, self.K_out // (n_jobs * 4))
        chunks = [list(range(i, min(i + chunk_size, self.K_out)))
                  for i in range(0, self.K_out, chunk_size)]
        
        if self.verbose:
            print(f"      Using {n_jobs} workers, {len(chunks)} chunks")
        
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_chunk)(chunk) for chunk in chunks
        )
        
        # Combine results
        all_rows = []
        all_cols = []
        all_vals = []
        for r, c, v in results:
            all_rows.extend(r)
            all_cols.extend(c)
            all_vals.extend(v)
        
        return np.array(all_rows, dtype=np.int64), np.array(all_cols, dtype=np.int64), np.array(all_vals, dtype=np.float64)

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
