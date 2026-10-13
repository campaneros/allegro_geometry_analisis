import os
import re
import sys
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyse import shape_vars, h68, GRID, OFFSET

RECO = "/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output/geometry_studies/ALLEGRO_strips/reco"
OUT = "/eos/user/m/mcampana/www/ALLEGRO/strip_layer/plots"
K4GEO = os.environ["K4GEO"]
DESIGNS = {"base": "ALLEGRO", "S3T": "ALLEGRO_S3T", "S4T": "ALLEGRO_S4T", "S5T": "ALLEGRO_S5T", "S4R": "ALLEGRO_B"}
W0 = [-1, 3.0, 3.0, 3.0, 4.25, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
NAMES = ["energy_fraction", "theta", "width_theta", "Ratio_E_max_2ndmax", "Delta_E_2ndmax_min", "width_theta_3Bin", "E_fr_side_pm3"]
A, C = "AugmentedEMBCaloClusters", "EMBCaloClusterCells"


def merging(geo):
    s = open(f"{K4GEO}/FCCee/{geo}/compact/ALLEGRO_o1_v03/ECalBarrel_thetamodulemerged.xml").read()
    return [int(x) for x in re.search(r'readout name="ECalBarrelModuleThetaMerged">.*?mergedCells_Theta="([\d ]+)"', s, re.S).group(1).split()]


def reco_like(E, tid, th, l, n_merge):
    El = E.sum()
    if W0[l] < 0:
        theta = (E * th).sum() / El
    else:
        w = np.maximum(0, W0[l] + np.log(E / El))
        theta = (w * th).sum() / w.sum() if w.sum() > 0 else 0.
    width = np.sqrt(max((E * tid ** 2).sum() / El - ((E * tid).sum() / El) ** 2, 0))
    ids = np.unique(tid)
    prof = np.array([E[tid == i].sum() for i in ids])
    full = np.arange(ids[0], ids[-1] + 1, n_merge)
    q = np.zeros(len(full))
    q[np.searchsorted(full, ids)] = prof
    m = len(q)
    left = np.r_[-np.inf, q[:-1]] if m > 1 else np.array([-np.inf])
    right = np.r_[q[1:], -np.inf] if m > 1 else np.array([-np.inf])
    peaks = np.where((q > left) & (q > right))[0]
    if len(peaks) == 0:
        return [El, theta, width, 1., 0., 0., 0.]
    order = peaks[np.argsort(q[peaks])[::-1]]
    i1 = order[0]
    e1, e2 = q[i1], (q[order[1]] if len(order) > 1 else 0.)
    emin = q[min(i1, order[1]) + 1:max(i1, order[1])].min() if len(order) > 1 and abs(i1 - order[1]) > 1 else 0.
    ratio = (e1 - e2) / (e1 + e2) if e1 + e2 > 0 else 1.
    get = lambda k: (q[k], full[k]) if 0 <= k < m else (0., 0)
    win = [get(i1 + k) for k in (-1, 0, 1)]
    s3 = sum(e for e, _ in win)
    s7 = s3 + sum(get(i1 + k)[0] for k in (-3, -2, 2, 3))
    w3 = np.sqrt(max(sum(e * t * t for e, t in win) / s3 - (sum(e * t for e, t in win) / s3) ** 2, 0)) if s3 > 0 else 0.
    return [El, theta, width, ratio, e2 - emin if len(order) > 1 else 0., w3, s7 / s3 - 1 if s3 > 0 else 0.]


def mine(E, tid, th, n_merge, u0):
    pos_cell = tid + (n_merge - 1) / 2 - u0
    keep = np.abs(pos_cell) <= 30
    E, pos_cell = E[keep], pos_cell[keep]
    centres = np.arange(-30, 31) if n_merge == 1 else np.unique(pos_cell)
    if n_merge > 1:
        lo, hi = centres.min(), centres.max()
        centres = np.arange(lo - n_merge * 8, hi + n_merge * 8 + 1, n_merge)
    q = np.zeros(len(centres))
    np.add.at(q, np.searchsorted(centres, pos_cell), E)
    q, pos = q[None], centres[None].astype(float)
    v = shape_vars(q, pos, n_merge)[0]
    c = pos[0, np.argmax(np.where(np.abs(pos[0]) <= 12, q[0], -1e9))]
    w = np.where(np.abs(pos[0] - c) <= (5 if n_merge == 1 else n_merge + 0.1), q[0], 0)
    theta = OFFSET + GRID * (u0 + (w * pos[0]).sum() / max(w.sum(), 1e-12))
    tot = np.where(np.abs(pos[0] - c) <= 16, q[0], 0).sum()
    return [E.sum(), theta, v[3], v[0], v[1] * tot, v[2] * n_merge, v[4]]


def process(design, part, E, nfiles=1):
    merge = merging(DESIGNS[design])
    nl = len(merge)
    rows = []
    for j in range(nfiles):
        f = f"{RECO}/{design}/theta90/{part}_{E}GeV_{j}_reco.root"
        br = [f"{A}/{A}.energy", f"{A}/{A}.hits_begin", f"{A}/{A}.hits_end", f"{A}/{A}.shapeParameters_begin",
              f"_{A}_hits/_{A}_hits.index", f"_{A}_shapeParameters", f"{C}/{C}.cellID", f"{C}/{C}.energy",
              f"{C}/{C}.position.x", f"{C}/{C}.position.y", f"{C}/{C}.position.z",
              "MCParticles/MCParticles.momentum.x", "MCParticles/MCParticles.momentum.y", "MCParticles/MCParticles.momentum.z"]
        a = uproot.open(f)["events"].arrays(br, library="np")
        for i in range(len(a[br[0]])):
            ecl = a[br[0]][i]
            if len(ecl) == 0:
                continue
            k = int(np.argmax(ecl))
            idx = a[br[4]][i][a[br[1]][i][k]:a[br[2]][i][k]]
            cid = a[br[6]][i][idx].astype(np.uint64)
            en = a[br[7]][i][idx]
            x, y, z = (a[b][i][idx] for b in br[8:11])
            th = np.arctan2(np.hypot(x, y), z)
            lay = ((cid >> np.uint64(11)) & np.uint64(0xFF)).astype(int)
            tid = ((cid >> np.uint64(30)) & np.uint64(0x3FF)).astype(int)
            sp = a[br[5]][i][a[br[3]][i][k]:]
            px, py, pz = (a[b][i][0] for b in br[11:14])
            tt = np.arctan2(np.hypot(px, py), pz)
            u0 = round((tt - OFFSET) / GRID)
            for l in range(1, nl):
                m = lay == l
                if not m.any():
                    continue
                r = sp[17 * l:17 * (l + 1)]
                reco = [r[0] * ecl[k], r[1], r[4], r[6], r[7], r[10], r[15]]
                rows.append([l, tt] + reco + reco_like(en[m], tid[m], th[m], l, 1) + mine(en[m], tid[m], th[m], merge[l], u0))
    return np.array(rows), nl


def main():
    os.makedirs(OUT, exist_ok=True)
    nv = len(NAMES)
    summary = []
    for part, E in [("gamma", 50), ("pi0", 50), ("gamma", 10)]:
        fig, axs = plt.subplots(3, nv, figsize=(4 * nv, 11), sharex=True)
        fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        for design in DESIGNS:
            d, nl = process(design, part, E)
            lay, tt = d[:, 0], d[:, 1]
            reco, copy, my = d[:, 2:2 + nv], d[:, 2 + nv:2 + 2 * nv], d[:, 2 + 2 * nv:]
            layers = np.arange(1, nl)
            for v, name in enumerate(NAMES):
                scale = 1e3 if name == "theta" else 1.
                dc = [(copy[lay == l, v] - reco[lay == l, v]) * scale for l in layers]
                dm = [(my[lay == l, v] - reco[lay == l, v]) * scale for l in layers]
                axs[0, v].plot(layers, [np.max(np.abs(x)) for x in dc], "o-", label=design)
                axs[1, v].plot(layers, [np.mean(x) for x in dm], "o-", label=design)
                axs[2, v].plot(layers, [h68(x) for x in dm], "o-", label=design)
                for l, x, y in zip(layers, dc, dm):
                    summary.append(f"{part} {E} {design} L{l} {name}: copy-reco max {np.max(np.abs(x)):.3g}, "
                                   f"mine-reco mean {np.mean(y):.3g} h68 {h68(y):.3g}")
            for col, (vals, lab) in enumerate([(reco[:, 1], "reco (log weights)"), (my[:, 1], "mine (linear)")]):
                res = [(vals[lay == l] - tt[lay == l]) * 1e3 for l in layers]
                axs2[0].plot(layers, [np.mean(x) for x in res], "o-" if col == 0 else "s--", label=f"{design}, {lab}")
                axs2[1].plot(layers, [h68(x) for x in res], "o-" if col == 0 else "s--", label=f"{design}, {lab}")
            print(part, E, design, "done", flush=True)
        for v, name in enumerate(NAMES):
            unit = " [mrad]" if name == "theta" else (" [GeV]" if name in ("energy_fraction", "Delta_E_2ndmax_min") else "")
            axs[0, v].set(title=name, yscale="log", ylabel="max |copy - reco|" + unit if v == 0 else None)
            axs[1, v].set(ylabel="mean (mine - reco)" + unit if v == 0 else None)
            axs[2, v].set(xlabel="design layer", ylabel="68% width of (mine - reco)" + unit if v == 0 else None)
            axs[1, v].axhline(0, color="k", lw=0.8)
        axs[0, 0].legend(fontsize=8)
        for ax in axs.ravel():
            ax.grid(alpha=0.3, which="both")
        fig.suptitle(f"{part} {E} GeV, theta = 90 deg, leading sliding-window cluster. Row 1: my copy of the reco formulas vs the reco "
                     "(validation). Rows 2-3: my analysis definitions vs the reco. energy_fraction compared as layer energy [GeV]")
        fig.tight_layout()
        fig.savefig(f"{OUT}/09_reco_compare_variables_{part}_{E}GeV.png", dpi=90)
        plt.close(fig)
        axs2[0].set(ylabel="mean (theta_layer - theta_true) [mrad]", xlabel="design layer")
        axs2[1].set(ylabel="68% width of (theta_layer - theta_true) [mrad]", xlabel="design layer", yscale="log")
        axs2[0].legend(fontsize=7, ncol=2)
        for ax in axs2:
            ax.grid(alpha=0.3, which="both")
        fig2.suptitle(f"theta per layer vs MC truth, {part} {E} GeV, theta = 90 deg (fixed impact point: no S-curve correction possible)")
        fig2.tight_layout()
        fig2.savefig(f"{OUT}/09_reco_compare_theta_vs_truth_{part}_{E}GeV.png", dpi=90)
        plt.close(fig2)
    open(f"{OUT}/09_reco_compare_summary.txt", "w").write("\n".join(summary) + "\n")
    print("done", OUT)


if __name__ == "__main__":
    main()
