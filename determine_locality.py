#!/usr/bin/env python3
"""
wavecar_localization.py

For every leaf directory (no subdirectories) under the given folders:
  1. find the WAVECAR (any prefix; plain or .gz),
  2. find the HOMO of each spin channel from the WAVECAR occupations,
  3. rebuild |psi(r)|^2 of the HOMO on a real-space grid (FFT of the plane-wave
     coefficients) and compute its inverse participation ratio (IPR),
  4. compare it with the median IPR of deeper, bulk-like valence bands,
  5. print: folder path, HOMO-up verdict, HOMO-down verdict, details.

The .gz file is read as a stream and never decompressed to disk.

Usage:
    python wavecar_localization.py DIR [DIR ...] [--threshold 3] [--out file.tsv] ...
"""

import argparse
import gzip
import os
import sys

import numpy as np

# Use scipy's multithreaded FFT if available; otherwise numpy's (single-threaded).
# Both give the same result.
try:
    from scipy import fft as _sfft

    def ifftn(a):
        return _sfft.ifftn(a, workers=-1)
except ImportError:
    ifftn = np.fft.ifftn

# ----------------------------------------------------------------------------
# Physical constants, same values VASP uses internally
# ----------------------------------------------------------------------------
RYTOEV = 13.605826                  # 1 Rydberg in eV
AUTOA = 0.529177249                 # 1 Bohr in Angstrom
HSQDTM = RYTOEV * AUTOA * AUTOA     # hbar^2 / (2 m_e) in eV*Angstrom^2  (~3.81)
TPI = 2.0 * np.pi


# ============================================================================
# WAVECAR reader
# ============================================================================
class WavecarReader:
    """Minimal reader for the standard binary WAVECAR (VASP 5.x/6.x).

    The file is a sequence of fixed-length records of RECL bytes:
      record 0           : RECL, ISPIN, RTAG                      (float64)
      record 1           : NKPTS, NBANDS, ENCUT, lattice A (3x3)   (float64)
      for spin s, k-point k:
        header record    : NPLW, kx, ky, kz, then (Re E, Im E, occ) per band
        NBANDS records   : NPLW plane-wave coefficients each (complex64/128)
    Record index of the (s, k) header = 2 + s*NKPTS*(NBANDS+1) + k*(NBANDS+1);
    band b (0-based) sits at header + 1 + b.

    Works on .gz files too: gzip supports seek(), and as long as offsets only
    increase it decompresses the stream once, without rewinding.
    """

    def __init__(self, path):
        self.path = path
        self.f = gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")
        try:
            self._read_global_header()
        except Exception:
            self.f.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.f.close()

    # --- low-level helpers ---------------------------------------------------
    def _read_at(self, offset, nbytes):
        """Seek to a byte offset and read exactly nbytes (error if truncated)."""
        self.f.seek(offset)
        buf = self.f.read(nbytes)
        if len(buf) != nbytes:
            raise EOFError(f"WAVECAR truncated (wanted {nbytes} bytes at offset {offset})")
        return buf

    def _read_doubles(self, offset, count):
        return np.frombuffer(self._read_at(offset, 8 * count), dtype="<f8")

    def _record_offset(self, ispin, ik, iband=None):
        """Byte offset of the (spin, k) header (iband=None) or of band iband."""
        rec = 2 + ispin * self.nkpts * (self.nbands + 1) + ik * (self.nbands + 1)
        if iband is not None:
            rec += 1 + iband
        return rec * self.recl

    # --- headers ---------------------------------------------------------------
    def _read_global_header(self):
        rec0 = self._read_doubles(0, 3)
        self.recl = int(rec0[0])        # record length in bytes
        self.nspin = int(rec0[1])       # ISPIN
        rtag = int(rec0[2])             # precision tag of the coefficients
        if rtag == 45200:
            self.cdtype = np.dtype("<c8")       # single-precision complex
        elif rtag == 45210:
            self.cdtype = np.dtype("<c16")      # double-precision complex
        else:
            raise ValueError(f"unsupported WAVECAR format tag {rtag}")

        rec1 = self._read_doubles(self.recl, 12)
        self.nkpts = int(rec1[0])
        self.nbands = int(rec1[1])
        self.encut = float(rec1[2])                 # eV
        self.A = rec1[3:12].reshape(3, 3).copy()    # rows = lattice vectors, Angstrom

    def read_kpoint_header(self, ispin, ik):
        """Return (nplw, kvec_fractional, eigenvalues, occupations) for (spin, k)."""
        d = self._read_doubles(self._record_offset(ispin, ik), 4 + 3 * self.nbands)
        nplw = int(round(d[0]))
        kvec = d[1:4].copy()
        table = d[4:].reshape(self.nbands, 3)       # columns: Re E, Im E, occupation
        return nplw, kvec, table[:, 0].copy(), table[:, 2].copy()

    def read_coefficients(self, ispin, ik, iband, nplw):
        """Plane-wave coefficients of one band, as complex128."""
        buf = self._read_at(self._record_offset(ispin, ik, iband), nplw * self.cdtype.itemsize)
        return np.frombuffer(buf, dtype=self.cdtype).astype(np.complex128)


# ============================================================================
# Plane-wave basis
# ============================================================================
def fft_box(A, encut):
    """Smallest integer box (per direction) holding all G with |G+k|^2 < ENCUT.

    sqrt(ENCUT/RYTOEV) = |G|_max in Bohr^-1 (in Rydberg units, E = |G|^2).
    TPI / (|a_i|/AUTOA) = reciprocal spacing along direction i in Bohr^-1.
    Their ratio = largest integer index needed. Same bound VASP and other
    WAVECAR readers use.
    """
    a_norm = np.linalg.norm(A, axis=1)
    cutof = np.ceil(np.sqrt(encut / RYTOEV) / (TPI / (a_norm / AUTOA))).astype(int)
    return 2 * cutof + 1        # indices -cutof..+cutof


def gvectors(A, encut, kvec, ngrid, half=None):
    """Integer G vectors inside the cutoff sphere, in VASP's storage order.

    VASP loops z (outer), y, x (inner), so x varies fastest. meshgrid with
    indexing='ij' over (z, y, x) followed by ravel reproduces that order.
    half='x' or 'z' keeps only half of G space (gamma-only build); filtering
    the full ordered list keeps the relative order.
    """
    axes = []
    for n in ngrid:
        v = np.arange(n)
        v[n // 2 + 1:] -= n               # FFT order: 0, 1, ..., c, -c, ..., -1
        axes.append(v)
    gz, gy, gx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    gx, gy, gz = gx.ravel(), gy.ravel(), gz.ravel()

    if half == "x":
        keep = (gx > 0) | ((gx == 0) & (gy > 0)) | ((gx == 0) & (gy == 0) & (gz >= 0))
    elif half == "z":
        keep = (gz > 0) | ((gz == 0) & (gy > 0)) | ((gz == 0) & (gy == 0) & (gx >= 0))
    else:
        keep = slice(None)
    g = np.stack([gx, gy, gz], axis=1)[keep]

    # Cartesian |G+k|: rows of inv(A).T are b_i / (2*pi), so (n + k) @ (2*pi*B) is in 1/Angstrom
    B = np.linalg.inv(A).T
    ekin = HSQDTM * np.sum(((g + kvec) @ (TPI * B)) ** 2, axis=1)   # kinetic energy, eV
    return g[ekin < encut]


def identify_basis(A, encut, kvec, nplw, ngrid, gamma_half):
    """Infer the VASP build from the number of stored plane waves.

    std : nplw == number of G in the full sphere
    ncl : nplw == 2x that (two spinor components) -> not supported here
    gam : nplw == number of G in the half sphere (only possible at k = 0)
    """
    g_full = gvectors(A, encut, kvec, ngrid)
    if nplw == len(g_full):
        return "std", g_full
    if nplw == 2 * len(g_full):
        raise NotImplementedError("noncollinear WAVECAR: spin-up/down HOMO is not defined")
    if np.allclose(kvec, 0.0):
        g_half = gvectors(A, encut, kvec, ngrid, half=gamma_half)
        if nplw == len(g_half):
            return "gam", g_half
    raise ValueError(f"stored plane waves ({nplw}) match no known basis "
                     f"(full sphere has {len(g_full)})")


def next_fast_len(n):
    """Smallest integer >= n whose only prime factors are 2, 3, 5 (fast FFT size)."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


# ============================================================================
# Real-space density and localization measure
# ============================================================================
def density_on_grid(coeffs, gvec, nfine, gamma, gamma_sqrt2):
    """|psi(r)|^2 on a real-space grid of shape nfine.

    The coefficients are placed at their G positions (negative indices wrap
    around), and an inverse FFT gives psi on the grid (Bloch phase e^{ikr}
    drops out of |psi|^2). nfine is >= 2x the coefficient box, so |psi|^2,
    whose Fourier components reach twice as far as psi's, is represented
    without aliasing. That makes the grid sums in the IPR exact.

    Gamma-only: psi is real, so c(-G) = conj(c(G)). Only half of G space is
    stored; the other half is filled by conjugation. If gamma_sqrt2, the stored
    G != 0 coefficients carry a factor sqrt(2) (so the half-set is normalized),
    which is removed first.
    """
    grid = np.zeros(tuple(nfine), dtype=np.complex128)
    i0 = gvec[:, 0] % nfine[0]
    i1 = gvec[:, 1] % nfine[1]
    i2 = gvec[:, 2] % nfine[2]

    if gamma:
        c = coeffs.copy()
        if gamma_sqrt2:
            g0 = np.all(gvec == 0, axis=1)       # the G = 0 entry
            c[~g0] /= np.sqrt(2.0)
        grid[i0, i1, i2] = c
        # -G partners; for G = 0 this rewrites conj(c0) = c0 (real for a real psi)
        grid[(-gvec[:, 0]) % nfine[0], (-gvec[:, 1]) % nfine[1], (-gvec[:, 2]) % nfine[2]] = np.conj(c)
    else:
        grid[i0, i1, i2] = coeffs

    psi = ifftn(grid)                            # overall scale is irrelevant (IPR is scale-free)
    return psi.real ** 2 + psi.imag ** 2


def ipr_and_pr_fraction(rho):
    """IPR = sum(rho^2) / (sum rho)^2, in [1/N, 1]; independent of normalization.
    PR_frac = 1 / (N * IPR), in (0, 1]: fraction of the cell volume the state
    effectively fills (1 = uniform)."""
    s1 = rho.sum()
    s2 = np.square(rho).sum()
    ipr = s2 / (s1 * s1)
    return ipr, 1.0 / (rho.size * ipr)


# ============================================================================
# Analysis of one WAVECAR
# ============================================================================
def analyze_wavecar(path, opts):
    """Return {'mode': 'std'|'gam', 'channels': [up, down]} where each channel is
    {'band', 'ratio', 'pr_frac', 'verdict'}. Raises on any problem."""
    if os.path.getsize(path) == 0:
        raise ValueError("file is empty (LWAVE = .FALSE.?)")

    with WavecarReader(path) as w:
        ik = opts.kpoint - 1
        if not 0 <= ik < w.nkpts:
            raise ValueError(f"k-point {opts.kpoint} requested, file has {w.nkpts}")

        ngrid = fft_box(w.A, w.encut)                         # coefficient box
        nfine = tuple(next_fast_len(2 * n) for n in ngrid)    # anti-aliased real-space grid

        mode, gvec = None, None
        channels = []
        for s in range(w.nspin):
            nplw, kvec, eig, occ = w.read_kpoint_header(s, ik)
            if gvec is None:                                  # same k for both spins -> reuse
                mode, gvec = identify_basis(w.A, w.encut, kvec, nplw, ngrid, opts.gamma_half)

            # HOMO of this channel: highest band with occupation > occ_tol
            occupied = np.flatnonzero(occ > opts.occ_tol)
            if occupied.size == 0:
                raise ValueError(f"no occupied bands in spin channel {s + 1}")
            homo = int(occupied.max())                        # 0-based

            # reference window: ref_count bands ending ref_offset below the HOMO
            ref_hi = homo - opts.ref_offset
            ref_lo = max(0, ref_hi - opts.ref_count + 1)
            ref = list(range(ref_lo, ref_hi + 1))             # empty if ref_hi < 0

            # read bands in ascending order -> only forward seeks (one pass through .gz)
            ipr = {}
            for b in sorted(set(ref) | {homo}):
                c = w.read_coefficients(s, ik, b, nplw)
                rho = density_on_grid(c, gvec, nfine, mode == "gam", opts.gamma_sqrt2)
                ipr[b] = ipr_and_pr_fraction(rho)

            ipr_homo, pr_homo = ipr[homo]
            if ref:
                ratio = ipr_homo / float(np.median([ipr[b][0] for b in ref]))
                verdict = "localized" if ratio >= opts.threshold else "delocalized"
            else:
                ratio, verdict = float("nan"), "n/a (no reference bands)"

            channels.append({"band": homo + 1,                # 1-based, as in VASP
                             "ratio": ratio,
                             "pr_frac": pr_homo,
                             "verdict": verdict})

        if w.nspin == 1:                                      # spin-degenerate: down = up
            channels.append(dict(channels[0]))

    return {"mode": mode, "channels": channels}


# ============================================================================
# Directory traversal and output
# ============================================================================
def find_leaves(top):
    """Yield (path, files) for every directory under top with no subdirectories."""
    for root, dirs, files in os.walk(top):
        dirs.sort()                         # deterministic traversal order
        if not dirs:
            yield root, files


def find_wavecars(files):
    """Files ending in 'WAVECAR' or 'WAVECAR.gz' (any prefix). If both X and X.gz
    exist, keep only the plain X (no decompression needed)."""
    cands = [f for f in files if f.endswith("WAVECAR") or f.endswith("WAVECAR.gz")]
    plain = {f for f in cands if f.endswith("WAVECAR")}
    return sorted(f for f in cands if not (f.endswith(".gz") and f[:-3] in plain))


HEADER = ["path", "HOMO_up", "HOMO_down", "file", "mode",
          "up_band", "up_ratio", "up_PRfrac",
          "down_band", "down_ratio", "down_PRfrac"]


def channel_columns(ch):
    return [ch["band"], f"{ch['ratio']:.2f}", f"{ch['pr_frac']:.4f}"]


def main():
    ap = argparse.ArgumentParser(description="HOMO localization (IPR) from VASP WAVECARs.")
    ap.add_argument("folders", nargs="+", help="top-level folders to search for leaf directories")
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="localized if IPR(HOMO)/median IPR(reference) >= this (default 3)")
    ap.add_argument("--ref-offset", type=int, default=10,
                    help="reference window ends this many bands below the HOMO (default 10)")
    ap.add_argument("--ref-count", type=int, default=10,
                    help="number of reference bands (default 10)")
    ap.add_argument("--occ-tol", type=float, default=0.1,
                    help="a band is occupied if occupation > this (default 0.1)")
    ap.add_argument("--kpoint", type=int, default=1, help="1-based k-point to analyze (default 1)")
    ap.add_argument("--gamma-half", choices=["x", "z"], default="x",
                    help="half-grid direction of vasp_gam builds (default x)")
    ap.add_argument("--no-gamma-sqrt2", dest="gamma_sqrt2", action="store_false",
                    help="do NOT remove the sqrt(2) factor from gamma-only G != 0 coefficients")
    ap.add_argument("--out", default="wavecar_localization.tsv", help="output TSV file")
    opts = ap.parse_args()
    if opts.ref_offset < 1 or opts.ref_count < 1:
        ap.error("--ref-offset and --ref-count must be >= 1")

    with open(opts.out, "w") as out:
        def emit(row):
            """Write one tab-separated row to stdout and to the output file."""
            line = "\t".join(str(x) for x in row)
            print(line, flush=True)
            out.write(line + "\n")
            out.flush()

        emit(HEADER)
        for top in opts.folders:
            top = os.path.abspath(top)
            if not os.path.isdir(top):
                emit([top, "not a directory", "not a directory"])
                continue
            for root, files in find_leaves(top):
                wavecars = find_wavecars(files)
                if not wavecars:
                    emit([root, "no WAVECAR", "no WAVECAR"])
                    continue
                for fname in wavecars:
                    fpath = os.path.join(root, fname)
                    print(f"reading {fpath}", file=sys.stderr, flush=True)
                    try:
                        res = analyze_wavecar(fpath, opts)
                    except Exception as e:      # one bad file must not stop the run
                        emit([root, "ERROR", "ERROR", fname, f"{type(e).__name__}: {e}"])
                        continue
                    up, dn = res["channels"]
                    emit([root, up["verdict"], dn["verdict"], fname, res["mode"],
                          *channel_columns(up), *channel_columns(dn)])


if __name__ == "__main__":
    main()
