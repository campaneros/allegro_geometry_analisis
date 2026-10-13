import json, math, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve

RED = "/eos/user/m/mcampana/ALLEGRO/analysis/strip_layer/reduced"
OUT = "/eos/user/m/mcampana/www/ALLEGRO/strip_layer/plots"
RECO = {"baseline": "/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output/baseline",
        "front": "/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output/solenoid_front_studies"}
GRID, OFFSET, HALF = 0.009817477 / 4, 0.5902785, 30
NB = 2 * HALF + 1
ENERGIES = [1, 2, 5, 10, 20, 35, 50, 80, 100]
EPID = [1, 2, 5, 10, 20, 35, 50, 80, 100]
CONFS = {"baseline": "solenoid behind ECAL", "front": "solenoid in front of ECAL (+0.81 $X_0$ Al)"}
X0 = {"baseline": np.array([0.31, 0.47, 2.33, 4.18, 6.17, 8.15, 10.13, 12.20, 14.27, 16.43, 18.63, 20.88])}
X0["front"] = X0["baseline"] + 0.81
XE = X0["baseline"] - X0["baseline"][0]
XLAB = "depth inside the ECAL [$X_0$] (upstream material excluded)"
DESIGNS = [("no strips", ())] + [(f"strips in layer {L}", (L,)) for L in range(1, 7)] + [("strips in layers 1-6", tuple(range(1, 7)))]
CANDS = [1, 2, 3, 4, 5, 6]
PIDLAYERS = [1, 2, 3, 4, 5, 6]
LCOL = dict(zip(CANDS, plt.cm.tab10(range(6))))
ECOL = lambda Es: dict(zip(Es, plt.cm.viridis(np.linspace(0, .9, len(Es)))))
rng = np.random.default_rng(1)
os.makedirs(OUT, exist_ok=True)
_cache = {}


def load(conf, part, E):
    k = (conf, part, E)
    if k not in _cache:
        d = np.load(f"{RED}/{conf}_{part}_{E}GeV.npz")
        _cache[k] = {k2: d[k2] for k2 in d.files}
    return _cache[k]


def depth(conf=None):
    return 0.5 * (XE[:-1] + XE[1:])


def to_cells(q, u0):
    shift = (4 - (u0 - HALF) % 4) % 4
    out = np.zeros(q.shape[:-1] + (NB // 4 + 2,)); cen = np.zeros_like(out)
    for j in range(NB // 4 + 2):
        lo = shift - 4 + 4 * j
        idx = lo[..., None] + np.arange(4)
        ok = (idx >= 0) & (idx < NB)
        out[..., j] = np.where(ok, np.take_along_axis(q, np.clip(idx, 0, NB - 1), -1), 0).sum(-1)
        cen[..., j] = lo + 1.5 - HALF
    return out, cen


def layer_profile(d, l, strip):
    q = d["prof"][:, l]
    if strip:
        return q, np.broadcast_to(np.arange(NB) - HALF, q.shape).astype(float), 1
    u0 = np.round((d["truth"][:, 0] - OFFSET) / GRID).astype(int)
    c, cen = to_cells(q, u0)
    return c, cen, 4


def shape_vars(q, pos, step):
    n, m = q.shape
    imax = np.argmax(np.where(np.abs(pos) <= 12, q, -1e9), 1)
    E1 = q[np.arange(n), imax]
    d = np.abs(pos - pos[np.arange(n), imax][:, None]) / step
    tot = np.where(d <= 16 / step, q, 0).sum(1) + 1e-9
    loc = np.zeros_like(q, bool)
    loc[:, 1:-1] = (q[:, 1:-1] > q[:, :-2]) & (q[:, 1:-1] >= q[:, 2:]) & (q[:, 1:-1] > 0.01 * tot[:, None])
    loc &= (d > 0) & (d <= 16 / step)
    E2 = np.where(loc, q, 0).max(1)
    i2 = np.argmax(np.where(loc, q, -1), 1)
    lo, hi = np.minimum(imax, i2), np.maximum(imax, i2)
    j = np.arange(m)[None]
    between = np.where((j > lo[:, None]) & (j < hi[:, None]), q, np.inf).min(1)
    dE = np.where(E2 > 0, E2 - np.where(np.isfinite(between), between, E2), 0)
    w = lambda r: np.where(d <= r, q, 0)
    bary = (w(16 / step) * pos).sum(1) / tot
    wtot = np.sqrt((w(16 / step) * (pos - bary[:, None]) ** 2).sum(1) / tot)
    c1, c3 = w(1.1).sum(1), w(3.1).sum(1)
    ws3 = np.sqrt((w(1.1) * d ** 2).sum(1) / (c1 + 1e-9))
    return np.stack([(E1 - E2) / (E1 + E2 + 1e-9), dE / tot, ws3, wtot, (c3 - c1) / (c1 + 1e-9), E2 / tot], 1)


VARS = ["Eratio", "dE", "w_s3", "w_tot", "f_side", "E2/E"]
DKINDS = [("no strips", 0, "cells")] + [(f"strips in layer {L}", L, "strip") for L in CANDS] + [("strips in layers 1-6", 0, "all")]


def has_strips(l, L, kind):
    return (kind == "all" and 1 <= l <= 6) or (kind == "strip" and l == L)


def feature_names(L, kind):
    n = [f"E frac L{l}" for l in range(7)]
    for l in PIDLAYERS:
        n += [f"{v} L{l} cells" for v in VARS]
        if has_strips(l, L, kind): n += [f"{v} L{l} strips" for v in VARS]
    return n


def design_features(d, L, kind):
    etot = d["prof"].sum((1, 2)) + 1e-9
    f = [d["prof"].sum(2)[:, :7] / etot[:, None]]
    for l in PIDLAYERS:
        f.append(shape_vars(*layer_profile(d, l, False)))
        if kind == "all" or (kind == "strip" and l == L):
            f.append(shape_vars(*layer_profile(d, l, True)))
    return np.concatenate(f, 1)


_bdt = {}


def bdt(conf, E, L, kind):
    k = (conf, E, L, kind)
    if k in _bdt: return _bdt[k]
    g, p = load(conf, "gamma", E), load(conf, "pi0", E)
    X = np.concatenate([design_features(g, L, kind), design_features(p, L, kind)])
    y = np.r_[np.zeros(len(g["prof"])), np.ones(len(p["prof"]))]
    tr = np.random.default_rng(7).random(len(y)) < 0.5
    s = np.zeros(len(y)); models = []
    for fold in (tr, ~tr):
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                             l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                             n_iter_no_change=20, random_state=1).fit(X[fold], y[fold])
        s[~fold] = clf.predict_proba(X[~fold])[:, 1]; models.append(clf)
    fpr, tpr, _ = roc_curve(1 - y, -s)
    cut = np.quantile(s[y == 0], 0.9)
    npi = (y == 1).sum(); eff = max((s[y == 1] <= cut).sum(), 1) / npi
    _bdt[k] = (1 / eff, math.sqrt((1 - eff) / (npi * eff)) / eff, (tpr, fpr),
               dict(score=s, y=y, cut=cut, X=X, model=models[0], test=~tr))
    return _bdt[k]


def gauss_core(x):
    mu, sg = np.median(x), 1.4826 * np.median(np.abs(x - np.median(x)))
    for _ in range(4):
        h, e = np.histogram(x, 60, (mu - 2 * sg, mu + 2 * sg)); c = 0.5 * (e[1:] + e[:-1])
        (a, mu, sg), cov = curve_fit(lambda t, a, m, s: a * np.exp(-0.5 * ((t - m) / s) ** 2), c, h, p0=[h.max(), mu, sg],
                                     sigma=np.sqrt(np.maximum(h, 1)))
        sg = abs(sg)
    return mu, sg, math.sqrt(cov[2, 2])


def sw_energy(conf, E):
    f = f"{RED}/{conf}_gamma_{E}GeV_swE.npy"
    if not os.path.exists(f):
        import uproot, awkward as ak
        b = "AugmentedEMBCaloClusters/AugmentedEMBCaloClusters.energy"
        a = uproot.open(f"{RECO[conf]}/stripseverywhere_reco_LArPb_10k_gamma_{E}GeV.root")["events"].arrays([b], library="ak")[b]
        np.save(f, ak.to_numpy(ak.fill_none(ak.max(a, axis=1), 0.)))
    return np.load(f)


def layer_residual(d, l, strip):
    q, pos, step = layer_profile(d, l, strip)
    utrue = (d["truth"][:, 0] - OFFSET) / GRID
    u0 = np.round(utrue)
    im = np.argmax(np.where(np.abs(pos) <= 12, q, -1e9), 1)
    c = pos[np.arange(len(q)), im]
    w = np.where(np.abs(pos - c[:, None]) <= (5 if strip else 4.1), q, 0)
    ub = u0 + (w * pos).sum(1) / (w.sum(1) + 1e-12)
    res = ub - utrue
    ev = np.arange(len(q)) % 2 == 0
    ib = np.clip((np.mod(ub + 0.5, step) / step * 20).astype(int), 0, 19)
    corr = np.array([np.median(res[ev & (ib == b)]) if (ev & (ib == b)).sum() > 5 else 0 for b in range(20)])
    return res - corr[ib]


def pointing(conf, E, L, kind):
    d = load(conf, "gamma", E); rho, th0 = d["rho"], d["truth"][:, 0]
    ev = np.arange(len(th0)) % 2 == 0
    Sw = Sx = Sy = Sxx = Sxy = 0.
    for l in range(1, 11):
        strip = (kind == "all" and l <= 6) or (kind == "strip" and l == L)
        dz = -layer_residual(d, l, strip) * GRID * rho[l] / np.sin(th0) ** 2
        ok = d["prof"][:, l].sum(1) > 0.01 * E
        if (ok & ev).sum() < 50: continue
        w = ok / h68(dz[ok & ev]) ** 2
        if l == L and kind == "strip":
            dzc = -layer_residual(d, l, False) * GRID * rho[l] / np.sin(th0) ** 2
            szL = (h68(dz[ok & ~ev]) * 10, h68(dzc[ok & ~ev]) * 10)
        z = rho[l] / np.tan(th0) + dz
        Sw, Sx, Sy, Sxx, Sxy = Sw + w, Sx + w * rho[l], Sy + w * z, Sxx + w * rho[l] ** 2, Sxy + w * rho[l] * z
    b = (Sw * Sxy - Sx * Sy) / (Sw * Sxx - Sx ** 2)
    z0 = (Sy - b * Sx) / Sw
    th = np.arctan2(1, b)
    if kind != "strip": szL = (np.nan, np.nan)
    return szL[0], szL[1], h68((th - th0)[~ev]) * 1e3, h68(z0[~ev]) * 10


def h68(x):
    return 0.5 * np.diff(np.nanpercentile(x, [16, 84]))[0]


def main():
    S = {}
    Es = [1, 5, 20, 50, 100]; ec = ECOL(Es)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    lf = {c: {E: load(c, "gamma", E)["prof"].sum(2) for E in Es} for c in CONFS}
    for ax, conf in zip(axs, CONFS):
        for E in Es:
            p = lf[conf][E]; f = p.mean(0) / p.sum(1).mean() * 100
            ax.step(XE, np.r_[f, f[-1]], where="post", color=ec[E], label=f"{E} GeV")
        for l in range(11): ax.text(depth(conf)[l], 0.3, str(l), ha="center", fontsize=8, color="gray")
        ax.set(title=f"{CONFS[conf]}", xlabel=XLAB + "  (gray: layer)", ylabel="mean energy fraction in the layer [%]",
               xlim=(0, 22), ylim=(0, 26))
        ax.grid(alpha=.3); ax.legend(title="$\\gamma$")
    for E in Es:
        r = (lf["front"][E].mean(0) / lf["front"][E].sum(1).mean()) / (lf["baseline"][E].mean(0) / lf["baseline"][E].sum(1).mean())
        axs[2].plot(range(1, 11), r[1:], "o-", color=ec[E], label=f"{E} GeV")
    axs[2].axhline(1, color="k", lw=.8)
    axs[2].set(title="ratio front / behind, same layer", xlabel="layer", ylabel="energy fraction ratio", xticks=range(1, 11), yscale="log")
    axs[2].grid(alpha=.3, which="both"); axs[2].legend()
    fig.suptitle("Longitudinal development of photon showers (sliding-window cluster cells)"); fig.tight_layout()
    fig.savefig(f"{OUT}/01_longitudinal_gamma.png", dpi=110); plt.close(fig)

    rows = [("gamma", 10), ("pi0", 10), ("gamma", 50), ("pi0", 50)]
    for conf in CONFS:
        fig, axs = plt.subplots(len(rows), len(DESIGNS), figsize=(4 * len(DESIGNS), 3.6 * len(rows)), sharey=True)
        for i, (part, E) in enumerate(rows):
            d = load(conf, part, E); rho = d["rho"]
            ev = np.where(d["truth"][:, 5] > 0.004)[0][3] if part == "pi0" else 3
            u0 = np.round((d["truth"][ev:ev + 1, 0] - OFFSET) / GRID).astype(int)
            vmax = np.log10(d["prof"][ev].max())
            for j, (name, strips) in enumerate(DESIGNS):
                ax = axs[i, j]
                for l in range(11):
                    if l in strips:
                        q = d["prof"][ev, l]; edges = np.arange(NB + 1) - HALF - 0.5
                    else:
                        c, cen = to_cells(d["prof"][ev:ev + 1, l], u0); q = c[0]; edges = np.r_[cen[0] - 2, cen[0][-1] + 2]
                    ax.pcolormesh(edges * GRID * rho[l], [l - 0.5, l + 0.5], np.log10(np.clip(q, 1e-5, None))[None],
                                  vmin=-4, vmax=vmax, cmap="viridis", shading="flat")
                ax.set(xlim=(-10, 10), ylim=(-0.5, 10.5), yticks=range(11))
                if i == 0: ax.set_title(name)
                if i == len(rows) - 1: ax.set_xlabel("z - z$_{axis}$ [cm]")
                if j == 0:
                    t = d["truth"][ev]
                    ax.set_ylabel(f"{part} {E} GeV\nlayer")
                    if part == "pi0": ax.text(-9.5, 10.3, f"E$_\\gamma$ = {t[3]:.1f}, {t[4]:.1f} GeV, $\\alpha$ = {t[5] * 1e3:.0f} mrad", color="w", fontsize=8, va="top")
        fig.suptitle(f"Single events seen by each readout design (strips only where stated, 4-strip cells elsewhere; colour = log10 E/GeV) - {CONFS[conf]}")
        fig.tight_layout(); fig.savefig(f"{OUT}/02_z_development_{conf}.png", dpi=90); plt.close(fig)

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (part, E) in zip(axs, [("gamma", 10), ("pi0", 10), ("gamma", 50), ("pi0", 50)]):
        m = {c: load(c, part, E)["prof"].mean(0) / load(c, part, E)["prof"].sum((1, 2)).mean() for c in CONFS}
        rho = load("baseline", part, E)["rho"]
        for l in range(1, 11):
            z = (np.arange(NB + 1) - HALF - 0.5) * GRID * rho[l]
            r = np.where(m["baseline"][l] > 1e-4, m["front"][l] / np.maximum(m["baseline"][l], 1e-12), np.nan)
            im = ax.pcolormesh(z, [l - 0.5, l + 0.5], r[None], cmap="RdBu_r", shading="flat",
                               norm=matplotlib.colors.LogNorm(0.5, 2))
        ax.set(xlim=(-8, 8), ylim=(0.5, 10.5), yticks=range(1, 11), xlabel="z - z$_{axis}$ [cm]", ylabel="layer",
               title=f"{part} {E} GeV")
    fig.colorbar(im, ax=axs, label="energy fraction, front / behind", ticks=[0.5, 0.7, 1, 1.4, 2], format="%.1f")
    fig.suptitle("Ratio of the mean z development, solenoid in front / behind ECAL, same layer and strip (white: <1e-4 of E)")
    fig.savefig(f"{OUT}/02_z_development_ratio.png", dpi=100); plt.close(fig)

    Es = [5, 20, 50, 100]; ec = ECOL(Es)
    W = {}
    for conf in CONFS:
        for E in Es:
            d = load(conf, "gamma", E); p, rho = d["prof"], d["rho"]
            x = (np.arange(NB) - HALF - d["ufrac"][:, None])[:, None, :]
            w = np.where(np.abs(x) <= 10, p, 0)
            rms = np.sqrt((w * x ** 2).sum(2) / (w.sum(2) + 1e-12))
            core = np.where(np.abs(x) <= 1.5, p, 0).sum(2) / (p.sum(2) + 1e-12)
            has = p.sum(2) > 0.005 * E
            W[conf, E] = (np.array([np.median(rms[has[:, l], l]) * GRID * rho[l] * 10 for l in range(11)]),
                          np.array([np.median(core[has[:, l], l]) for l in range(11)]))
    for xname in ("X0", "layer"):
        fig, axs = plt.subplots(2, 3, figsize=(20, 9))
        xv = depth()[1:] if xname == "X0" else np.arange(1, 11)
        for j, conf in enumerate(CONFS):
            for E in Es:
                axs[0, j].plot(xv, W[conf, E][0][1:], "o-", color=ec[E], label=f"{E} GeV")
                axs[1, j].plot(xv, W[conf, E][1][1:], "o-", color=ec[E])
            if xname == "X0":
                for l in range(1, 11): axs[1, j].text(depth()[l], 0.12, str(l), ha="center", fontsize=8, color="gray")
            axs[0, j].axhline(GRID * 223 * 10, color="k", ls=":"); axs[0, j].text(xv[0], GRID * 223 * 10 + .3, "strip pitch", fontsize=8)
            axs[0, j].set(title=CONFS[conf], ylabel="median lateral RMS in z [mm]", ylim=(0, 28))
            axs[1, j].set(xlabel=(XLAB + "  (gray: layer)") if xname == "X0" else "layer",
                          ylabel="median fraction of the layer energy\nin the 3 strips around the photon axis", ylim=(0.1, 1))
            axs[0, j].legend(title="$\\gamma$")
        for E in Es:
            axs[0, 2].plot(xv, W["front", E][0][1:] / W["baseline", E][0][1:], "o-", color=ec[E])
            axs[1, 2].plot(xv, W["front", E][1][1:] / W["baseline", E][1][1:], "o-", color=ec[E])
        axs[0, 2].set(title="ratio front / behind, same layer", ylabel="RMS ratio")
        axs[1, 2].set(xlabel=XLAB if xname == "X0" else "layer", ylabel="core fraction ratio")
        for a in axs.ravel(): a.grid(alpha=.3)
        for a in axs[:, 2]: a.axhline(1, color="k", lw=.8)
        if xname == "layer":
            for a in axs.ravel(): a.set_xticks(range(1, 11))
        fig.suptitle("Lateral (z) size of photon showers in each layer at strip granularity (= what the strip layer would measure if placed there);"
                     " events with >0.5% of E in the layer")
        fig.tight_layout(); fig.savefig(f"{OUT}/03_lateral_width_vs_{xname}.png", dpi=110); plt.close(fig)

    Es = [1, 5, 20, 50, 100]; ec = ECOL(Es)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, conf in zip(axs, CONFS):
        for E in Es:
            p = lf_all = load(conf, "gamma", E)["prof"].sum(2)
            cum = np.cumsum(p, 1) / E
            prob = [np.mean(cum[:, L - 1] < 0.01) for L in range(1, 7)]
            ax.plot(XE[1:7], prob, "o-", color=ec[E], label=f"{E} GeV")
            for L in range(1, 7): ax.text(XE[L], 1.3, f"L{L}", ha="center", fontsize=8, color="gray")
        t = np.linspace(0, 12, 50); ax.plot(t, np.exp(-7 / 9 * (t + X0[conf][0])), "k:", label="exp(-7/9 t), t from the IP")
        ax.set(title=CONFS[conf], xlabel="depth of the front of the layer inside the ECAL [$X_0$]", yscale="log", ylim=(1e-3, 2), xlim=(0, 12))
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    axs[0].set_ylabel("fraction of photons with < 1% of E deposited before the layer")
    fig.suptitle("Photons still unconverted when they reach layer L (they leave little or no signal in L)")
    fig.tight_layout(); fig.savefig(f"{OUT}/04_unconverted_photons.png", dpi=110); plt.close(fig)

    S["rejection"] = {"E": EPID}
    fig, axs = plt.subplots(1, 3, figsize=(21, 5.5))
    for ax, conf in zip(axs, CONFS):
        S["rejection"][conf] = {}
        for L in CANDS:
            r = [bdt(conf, E, L, "strip")[0] for E in EPID]; S["rejection"][conf][f"L{L}"] = r
            ax.errorbar(EPID, r, [bdt(conf, E, L, "strip")[1] for E in EPID], fmt="o-", color=LCOL[L], capsize=2,
                        label=f"strips only in layer {L} ({XE[L]:.1f}-{XE[L + 1]:.1f} $X_0$ in ECAL)")
            print(conf, "L", L, np.round(r, 1), flush=True)
        for kind, st, lab in [("cells", "k:", "no strips (cells only)"), ("all", "k--", "strips in layers 1-6, cells in 7-10")]:
            r = [bdt(conf, E, 0, kind)[0] for E in EPID]; S["rejection"][conf][kind] = r
            ax.errorbar(EPID, r, [bdt(conf, E, 0, kind)[1] for E in EPID], fmt=st, capsize=2, label=lab)
            print(conf, kind, np.round(r, 1), flush=True)
        ax.set(xscale="log", yscale="log", xlabel="E [GeV]", ylabel="$\\pi^0$ rejection at 90% $\\gamma$ efficiency", title=CONFS[conf], ylim=(1, 20))
        ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    for L in CANDS:
        axs[2].plot(EPID, np.array(S["rejection"]["front"][f"L{L}"]) / np.array(S["rejection"]["baseline"][f"L{L}"]), "o-", color=LCOL[L], label=f"layer {L}")
    axs[2].axhline(1, color="k", lw=.8)
    axs[2].set(xscale="log", xlabel="E [GeV]", ylabel="rejection ratio front / behind", title="ratio, strips in the same layer")
    axs[2].grid(alpha=.3, which="both"); axs[2].legend(fontsize=8)
    fig.suptitle("$\\gamma/\\pi^0$ BDT (2-fold cross-validated): lateral shapes of layers 1-6 (4-strip cells, + strip shapes where the design has strips) + longitudinal fractions, no noise\n"
                 "1-2 GeV: the two photons are mostly further apart than the cluster window, the shape BDT is not the right tool there")
    fig.tight_layout(); fig.savefig(f"{OUT}/05_pi0_rejection.png", dpi=110); plt.close(fig)
    for conf in CONFS:
        fig, axs = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
        for ax, E in zip(axs, [10, 20, 50, 100]):
            for L in CANDS:
                tpr, fpr = bdt(conf, E, L, "strip")[2]
                ax.plot(tpr, 1 / np.maximum(fpr, 1e-4), color=LCOL[L], label=f"strips in layer {L}")
            for kind, st, lab in [("cells", "k:", "no strips"), ("all", "k--", "strips in 1-6")]:
                tpr, fpr = bdt(conf, E, 0, kind)[2]; ax.plot(tpr, 1 / np.maximum(fpr, 1e-4), st, label=lab)
            ax.axvline(0.9, color="gray", lw=.8)
            ax.set(title=f"{E} GeV", xlabel="$\\gamma$ efficiency", xlim=(0.5, 1.0), yscale="log", ylim=(1, 100))
            ax.grid(alpha=.3, which="both")
        axs[0].set_ylabel("$\\pi^0$ rejection (1/$\\pi^0$ efficiency)"); axs[0].legend(fontsize=8)
        fig.suptitle(f"ROC curves of the $\\gamma/\\pi^0$ BDT - {CONFS[conf]}"); fig.tight_layout()
        fig.savefig(f"{OUT}/06_roc_{conf}.png", dpi=110); plt.close(fig)

    EPT = [1, 2, 5, 10, 20, 35, 50, 80, 100]
    S["pointing"] = {"E": EPT}
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for j, conf in enumerate(CONFS):
        S["pointing"][conf] = {}
        for L, kind in [(L, "strip") for L in CANDS] + [(0, "cells"), (0, "all")]:
            r = np.array([pointing(conf, E, L, kind) for E in EPT])
            st = dict(color=LCOL[L], marker="o", label=f"strips only in layer {L}") if kind == "strip" else \
                dict(color="k", ls=":" if kind == "cells" else "--", label="no strips (all 4-strip cells)" if kind == "cells" else "strips in layers 1-6, cells in 7-10")
            for i in (0, 1): axs[i, j].plot(EPT, r[:, i + 2], **st)
            S["pointing"][conf][f"L{L}" if kind == "strip" else kind] = r[:, 2:].tolist()
            print(conf, "pointing", L, kind, np.round(r[:, 2:], 1).tolist(), flush=True)
        axs[0, j].set(title=CONFS[conf], ylabel="$\\sigma_\\theta$: 68% half-width of $\\theta_{reco}-\\theta_{true}$ [mrad]", yscale="log")
        axs[1, j].set(xlabel="E$_\\gamma$ [GeV]", ylabel="$\\sigma_{z_0}$: 68% half-width of z at the beam line\n(true vertex z = 0) [mm]",
                      xscale="log", yscale="log")
        for a in axs[:, j]: a.grid(alpha=.3, which="both")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Photon pointing from the cluster: weighted line fit to the S-corrected barycentres of layers 1-10\n"
                 "(each layer read with the granularity of the design: strips where stated, 4-strip cells elsewhere; no noise)")
    fig.tight_layout(); fig.savefig(f"{OUT}/07_pointing.png", dpi=110); plt.close(fig)

    Es = [20, 50]
    dcol = dict(zip([n for n, _, _ in DKINDS], ["k"] + [LCOL[L] for L in CANDS] + ["gray"]))
    dls = lambda kind: ":" if kind == "cells" else ("--" if kind == "all" else "-")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for i, conf in enumerate(CONFS):
        for j, E in enumerate(Es):
            d = load(conf, "gamma", E); rho, th0 = d["rho"], d["truth"][:, 0]
            ev = np.arange(len(th0)) % 2 == 0
            for name, L, kind in DKINDS:
                rms, sz = [], []
                for l in range(1, 11):
                    q, pos, step = layer_profile(d, l, has_strips(l, L, kind))
                    im = np.argmax(np.where(np.abs(pos) <= 12, q, -1e9), 1); c = pos[np.arange(len(q)), im]
                    w = np.where(np.abs(pos - c[:, None]) <= 16, q, 0); tot = w.sum(1) + 1e-12
                    b = (w * pos).sum(1) / tot
                    ok = d["prof"][:, l].sum(1) > 0.005 * E
                    rms.append(np.median(np.sqrt((w * (pos - b[:, None]) ** 2).sum(1) / tot)[ok]) * GRID * rho[l] * 10)
                    dz = -layer_residual(d, l, has_strips(l, L, kind)) * GRID * rho[l] / np.sin(th0) ** 2
                    sz.append(h68(dz[ok & ~ev]) * 10)
                axs[i, j].plot(range(1, 11), rms, ls=dls(kind), marker="o", color=dcol[name], label=name)
                axs2[i, j].plot(range(1, 11), sz, ls=dls(kind), marker="o", color=dcol[name], label=name)
            axs[i, j].set(title=f"{CONFS[conf]}, $\\gamma$ {E} GeV", ylabel="median measured RMS in z [mm]")
            axs2[i, j].set(title=f"{CONFS[conf]}, $\\gamma$ {E} GeV", ylabel="z resolution of the layer barycentre [mm]", yscale="log")
    for a in list(axs.ravel()) + list(axs2.ravel()): a.grid(alpha=.3, which="both"); a.set_xticks(range(1, 11))
    for a in list(axs[1]) + list(axs2[1]): a.set_xlabel("layer")
    axs[0, 0].legend(fontsize=8); axs2[0, 0].legend(fontsize=8)
    fig.suptitle("Lateral width measured by each design (strips where the design has them, 4-strip cells elsewhere), around the reconstructed barycentre")
    fig.tight_layout(); fig.savefig(f"{OUT}/03b_measured_width_per_design.png", dpi=110); plt.close(fig)
    fig2.suptitle("z resolution of the S-corrected barycentre of each layer, per design (the input of the pointing fit)")
    fig2.tight_layout(); fig2.savefig(f"{OUT}/03c_layer_position_resolution_per_design.png", dpi=110); plt.close(fig2)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for i, conf in enumerate(CONFS):
        for j, part in enumerate(["pi0", "gamma"]):
            for name, L, kind in DKINDS:
                fr = []
                for E in EPID:
                    d = load(conf, part, E); two = np.zeros(len(d["prof"]), bool)
                    for l in range(1, 7):
                        v = shape_vars(*layer_profile(d, l, has_strips(l, L, kind)))
                        two |= v[:, 0] < (1 - 0.1) / (1 + 0.1)
                    fr.append(two.mean())
                axs[i, j].plot(EPID, fr, ls=dls(kind), marker="o", color=dcol[name], label=name)
            what = "$\\pi^0$ (wanted)" if part == "pi0" else "$\\gamma$ (fakes)"
            axs[i, j].set(title=f"{CONFS[conf]}: {what}", xscale="log",
                          ylabel="fraction of events with two peaks", ylim=(0, 1))
            axs[i, j].grid(alpha=.3, which="both")
    for a in axs[1]: a.set_xlabel("E [GeV]")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Two separate maxima seen in layers 1-6 (second local maximum > 10% of the first, within 16 strips), per design")
    fig.tight_layout(); fig.savefig(f"{OUT}/03d_two_peaks_per_design.png", dpi=110); plt.close(fig)

    from sklearn.inspection import permutation_importance
    SHOW = [DKINDS[0], DKINDS[1], DKINDS[2], DKINDS[3], DKINDS[4], DKINDS[-1]]
    for conf in CONFS:
        Eshow = [10, 20, 50, 100]
        fig, axs = plt.subplots(len(SHOW), len(Eshow), figsize=(5 * len(Eshow), 2.8 * len(SHOW)), sharex=True)
        for i, (name, L, kind) in enumerate(SHOW):
            for j, E in enumerate(Eshow):
                dt = bdt(conf, E, L, kind)[3]; b = np.linspace(0, 1, 51)
                axs[i, j].hist(dt["score"][dt["y"] == 0], b, histtype="step", density=True, label="$\\gamma$")
                axs[i, j].hist(dt["score"][dt["y"] == 1], b, histtype="step", density=True, label="$\\pi^0$")
                axs[i, j].axvline(dt["cut"], color="k", ls=":"); axs[i, j].set_yscale("log")
                axs[i, j].set_title(f"{name}, {E} GeV", fontsize=9)
        for a in axs[-1]: a.set_xlabel("BDT score ($\\pi^0$-like)")
        axs[0, 0].legend()
        fig.suptitle(f"BDT score (cross-validated), dotted: cut at 90% $\\gamma$ efficiency - {CONFS[conf]}")
        fig.tight_layout(); fig.savefig(f"{OUT}/06b_bdt_score_{conf}.png", dpi=90); plt.close(fig)

        for E in (20, 50):
            g, p = load(conf, "gamma", E), load(conf, "pi0", E)
            fig, axs = plt.subplots(7, 6, figsize=(24, 22))
            for i, L in enumerate(range(1, 7)):
                vg, vp = shape_vars(*layer_profile(g, L, True)), shape_vars(*layer_profile(p, L, True))
                for k in range(6):
                    lo, hi = np.percentile(np.r_[vg[:, k], vp[:, k]], [0.5, 99.5]); b = np.linspace(lo, hi + 1e-9, 50)
                    axs[i, k].hist(vg[:, k], b, histtype="step", density=True, label="$\\gamma$")
                    axs[i, k].hist(vp[:, k], b, histtype="step", density=True, label="$\\pi^0$")
                    axs[i, k].set_title(f"{VARS[k]}, strips in layer {L}", fontsize=9)
            fl = lambda d: d["prof"].sum(2)[:, :6] / (d["prof"].sum((1, 2))[:, None] + 1e-9)
            for k in range(6):
                b = np.linspace(0, max(np.percentile(fl(g)[:, k], 99.5), 1e-3), 50)
                axs[6, k].hist(fl(g)[:, k], b, histtype="step", density=True, label="$\\gamma$")
                axs[6, k].hist(fl(p)[:, k], b, histtype="step", density=True, label="$\\pi^0$")
                axs[6, k].set_title(f"energy fraction in layer {k} (all designs)", fontsize=9)
            axs[0, 0].legend()
            fig.suptitle(f"BDT input variables, $\\gamma$ vs $\\pi^0$ at {E} GeV - {CONFS[conf]}\n"
                         "Eratio=(E1-E2)/(E1+E2); dE=valley depth/E; w_s3=width in +-1 channel; w_tot=width in +-16 strips; f_side=(E+-3-E+-1)/E+-1; E2/E")
            fig.tight_layout(); fig.savefig(f"{OUT}/06c_bdt_inputs_{conf}_{E}GeV.png", dpi=80); plt.close(fig)

        fig, axs = plt.subplots(2, len(SHOW), figsize=(5 * len(SHOW), 10))
        for i, E in enumerate((20, 50)):
            for j, (name, L, kind) in enumerate(SHOW):
                dt = bdt(conf, E, L, kind)[3]
                pi = permutation_importance(dt["model"], dt["X"][dt["test"]], dt["y"][dt["test"]], scoring="roc_auc", n_repeats=3, random_state=1)
                o = np.argsort(pi.importances_mean)[::-1][:12]
                nm = feature_names(L, kind)
                axs[i, j].barh(range(len(o))[::-1], pi.importances_mean[o], xerr=pi.importances_std[o], color="C0")
                axs[i, j].set_yticks(range(len(o))[::-1]); axs[i, j].set_yticklabels([nm[k] for k in o], fontsize=8)
                axs[i, j].set_title(f"{name}, {E} GeV", fontsize=9); axs[i, j].set_xlabel("AUC loss when shuffled")
        fig.suptitle(f"BDT permutation importance (12 most important inputs) - {CONFS[conf]}")
        fig.tight_layout(); fig.savefig(f"{OUT}/06d_bdt_importance_{conf}.png", dpi=90); plt.close(fig)

    S["energy"] = {}
    fig, axs = plt.subplots(1, 4, figsize=(26, 5.5))
    res = {}
    for conf in CONFS:
        r = []
        for E in ENERGIES:
            e = sw_energy(conf, E); e = e[e > 0.3 * E]
            mu, sg, dsg = gauss_core(e / E)
            r.append((mu, sg / mu, dsg / mu))
        res[conf] = np.array(r)
        (a, c), _ = curve_fit(lambda E, a, c: np.sqrt(a ** 2 / E + c ** 2), ENERGIES, res[conf][:, 1], sigma=res[conf][:, 2], p0=[0.08, 0.005])
        S["energy"][conf] = {"E": ENERGIES, "response": res[conf][:, 0].tolist(), "sigma_over_E": res[conf][:, 1].tolist(),
                             "stochastic": abs(a), "constant": abs(c)}
        axs[0].errorbar(ENERGIES, res[conf][:, 0], fmt="o-", label=CONFS[conf])
        axs[1].errorbar(ENERGIES, res[conf][:, 1] * 100, res[conf][:, 2] * 100, fmt="o",
                        label=f"{CONFS[conf]}: {abs(a) * 100:.1f}%/$\\sqrt{{E}}$ $\\oplus$ {abs(c) * 100:.2f}%", color=axs[0].lines[-1].get_color())
        t = np.logspace(0, 2, 50); axs[1].plot(t, np.sqrt(a ** 2 / t + c ** 2) * 100, color=axs[0].lines[-1].get_color())
        print(conf, "sigma/E", np.round(res[conf][:, 1] * 100, 2), "a, c", round(abs(a), 4), round(abs(c), 4), flush=True)
    axs[2].plot(ENERGIES, res["front"][:, 1] / res["baseline"][:, 1], "o-")
    axs[2].axhline(1, color="k", lw=.8)
    axs[0].set(xscale="log", xlabel="E$_\\gamma$ [GeV]", ylabel="response E$_{SW}$/E (Gaussian core)", title="response (no upstream correction)")
    axs[1].set(xscale="log", yscale="log", xlabel="E$_\\gamma$ [GeV]", ylabel="$\\sigma_E$/E [%]", title="resolution")
    axs[2].set(xscale="log", xlabel="E$_\\gamma$ [GeV]", ylabel="$\\sigma_E$/E ratio front / behind", title="ratio")
    for conf, ls in [("baseline", "-"), ("front", "--")]:
        for L in [None] + CANDS:
            r = []
            for E in ENERGIES:
                d = load(conf, "gamma", E); u0 = np.round((d["truth"][:, 0] - OFFSET) / GRID).astype(int)
                c, cen = to_cells(d["prof"].sum(1), u0); lo = cen[np.arange(len(c)), np.argmax(c, 1)] - 18
                tot = 0
                for l in range(11):
                    q, pos, _ = layer_profile(d, l, l == L)
                    tot = tot + np.where((pos >= lo[:, None]) & (pos < lo[:, None] + 36), q, 0).sum(1)
                r.append(gauss_core(tot / E)[1])
            if L is None: ref = np.array(r); continue
            axs[3].plot(ENERGIES, np.array(r) / ref, ls, marker="o", color=LCOL[L], label=f"strips in layer {L}" if conf == "baseline" else None)
    axs[3].set(xscale="log", xlabel="E$_\\gamma$ [GeV]", ylabel="$\\sigma_E$/E (strips in layer L) / $\\sigma_E$/E (cells only)",
               title="designs, 9-tower window (solid: behind, dashed: front)", ylim=(0.99, 1.01))
    axs[3].legend(fontsize=8)
    for a in axs: a.grid(alpha=.3, which="both")
    axs[0].legend(fontsize=8); axs[1].legend(fontsize=8)
    fig.suptitle("Photon energy, leading sliding-window cluster. Without noise the cluster energy does not depend on where the strips are:"
                 " the same energy is only regrouped in theta")
    fig.tight_layout(); fig.savefig(f"{OUT}/08_energy_resolution.png", dpi=110); plt.close(fig)

    json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    print("done", OUT)


if __name__ == "__main__":
    main()
