#!/usr/bin/env python3
"""Add compute_bispectrum to the encoder (idempotent) and run an OFFLINE de-aliasing test on KITTI.
Tests the bispectrum gamble BEFORE any cache rebuild / retrain:
 - alias pairs = distant places (>=GAMMA) that collide in the magnitude key d (<=EPS)
 - does the bicoherence bispectrum SEPARATE those pairs? (mean bispec distance for alias pairs
   vs random non-colliding distant pairs) -> if alias pairs are separable in bispec, it de-aliases.
 - also: AliasRate on d alone vs on [d ; bispec] (concatenated, both L2-normalized).
"""
import sys, os
sys.path.insert(0, "src")
import numpy as np, torch

ENC = "src/encoding/spectral_encoder.py"

METHOD = '''
    def compute_bispectrum(self, points):
        """Compact low-frequency bicoherence bispectrum: yaw-shift-invariant phase-coupling
        recovered from the COMPLEX per-row rfft (shift phases cancel exactly). Complements the
        magnitude key with the phase-coupling info magnitude discards. Returns (rows*2*n_pairs,)."""
        image_2d, _ = self.projector.project(points, keep_intensity=False)
        if self.interpolate_empty:
            if self.projection_type == "bev":
                image_2d = interpolate_bev_image(image_2d, method="linear", n_channels=self.bev_n_channels)
            else:
                image_2d = interpolate_range_image(image_2d, method="linear")
        image_tensor = torch.from_numpy(image_2d).float().to(self.alpha.device)
        if self.projection_type != "bev" and image_tensor.shape[0] != self.target_elevation_bins:
            image_tensor = torch.nn.functional.adaptive_avg_pool2d(
                image_tensor.unsqueeze(0).unsqueeze(0),
                (self.target_elevation_bins, image_tensor.shape[1])).squeeze()
        if self.zero_center:
            image_tensor = image_tensor - image_tensor.mean(dim=1, keepdim=True)
        X = torch.fft.rfft(image_tensor, dim=1, norm="ortho")
        nf = X.shape[1]
        pairs = [(k1, k2) for k1 in range(1, 6) for k2 in range(1, k1 + 1) if k1 + k2 <= 6 and k1 + k2 < nf]
        feats = []
        for (k1, k2) in pairs:
            B = X[:, k1] * X[:, k2] * torch.conj(X[:, k1 + k2])
            denom = X[:, k1].abs() * X[:, k2].abs() * X[:, k1 + k2].abs() + self.epsilon
            bic = B / denom
            feats.append(bic.real); feats.append(bic.imag)
        bispec = torch.stack(feats, dim=-1)
        return bispec.reshape(-1).detach().cpu().numpy().astype(np.float32)
'''

def patch_encoder():
    s = open(ENC).read()
    if "def compute_bispectrum" in s:
        print("compute_bispectrum already present"); return
    anchor = "        return fft_magnitudes.detach().cpu().numpy().astype(np.float32)\n"
    assert anchor in s, "anchor not found"
    i = s.index(anchor) + len(anchor)
    s = s[:i] + METHOD + s[i:]
    open(ENC, "w").write(s)
    print("patched compute_bispectrum")

def main():
    patch_encoder()
    import ast; ast.parse(open(ENC).read()); print("syntax OK")
    from data.kitti_loader import KITTILoader
    from run_kitti_operating_point import _make_encoder
    from evaluate_dump import _apply_encoder_preset, _load_config
    cfg = _apply_encoder_preset(_load_config("configs/training_kitti_phase_alignment_gat_fast.yaml"), "no_interdiff")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = _make_encoder(cfg, dev)
    root = "/rise/RISE1/workspace/data/kitti/dataset"
    seq = "08"
    loader = KITTILoader(root, seq, lazy_load=True)
    N = len(loader)
    # sample every STRIDE scan to keep it fast (~800 frames)
    STRIDE = max(1, N // 800)
    idxs = list(range(0, N, STRIDE))
    D, BS, POS = [], [], []
    for k, i in enumerate(idxs):
        it = loader[i]
        D.append(enc.encode_points(it["points"]).detach().cpu().numpy().astype(np.float64))
        BS.append(enc.compute_bispectrum(it["points"]).astype(np.float64))
        POS.append(np.asarray(it["pose"])[:3, 3])
        if k % 100 == 0:
            print(f"  {k}/{len(idxs)} bispec_dim={len(BS[-1])}", flush=True)
    D = np.array(D); BS = np.array(BS); POS = np.array(POS)
    print("shapes d", D.shape, "bispec", BS.shape, "pos", POS.shape)

    def l2(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    dn = l2(D); bn = l2(BS)
    GAMMA, EPS = 25.0, 0.1
    rng = np.random.RandomState(0)
    M = len(dn)
    a = rng.randint(0, M, 200000); b = rng.randint(0, M, 200000)
    m = a != b; a, b = a[m], b[m]
    far = np.linalg.norm(POS[a] - POS[b], axis=1) >= GAMMA
    a, b = a[far], b[far]
    dd = np.linalg.norm(dn[a] - dn[b], axis=1)
    alias = dd <= EPS               # distant places colliding in magnitude key = spectral aliases
    nonalias = dd > EPS
    print(f"\n=== KITTI08 de-aliasing test: {alias.sum()} alias pairs / {len(a)} distant pairs ===")
    if alias.sum() > 0:
        bd_alias = np.linalg.norm(bn[a[alias]] - bn[b[alias]], axis=1)
        bd_non = np.linalg.norm(bn[a[nonalias]] - bn[b[nonalias]], axis=1)
        print(f"  bispec distance  alias-pairs mean={bd_alias.mean():.4f}  non-alias distant mean={bd_non.mean():.4f}")
        print(f"  => alias pairs (identical in d) have bispec distance {bd_alias.mean():.3f}; if >>0 the bispectrum SEPARATES them (de-aliases)")
        # fraction of alias pairs that bispec pushes apart beyond a threshold
        for th in [0.1, 0.2, 0.3]:
            print(f"     alias pairs with bispec-dist>{th}: {(bd_alias>th).mean()*100:.1f}%")
    # AliasRate: d vs [d;bispec]
    def aliasrate(z):
        dz = np.linalg.norm(z[a] - z[b], axis=1)
        return (dz <= EPS).mean()
    joint = l2(np.concatenate([dn, bn], axis=1))
    print(f"\n  AliasRate(d)={aliasrate(dn)*100:.2f}%   AliasRate([d;bispec])={aliasrate(joint)*100:.2f}%")

if __name__ == "__main__":
    main()
