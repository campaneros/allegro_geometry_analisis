import os
from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve

RED = "/eos/user/m/mcampana/ALLEGRO/analysis/strip_layer/reduced"
OUT = "/eos/user/m/mcampana/www/ALLEGRO/strip_layer/plots"
BASE = "/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output"
RECO = {"baseline": f"{BASE}/baseline", "front": f"{BASE}/solenoid_front_studies"}
GRID, OFFSET, HALF = 0.009817477 / 4, 0.5902785, 30
NB = 2 * HALF + 1
ENERGIES = [1, 2, 5, 10, 20, 35, 50, 80, 100]
CONFS = {"baseline": "solenoid behind ECAL", "front": "solenoid in front of ECAL (+0.81 $X_0$ Al)"}
UPSTREAM = {"baseline": 0.31, "front": 1.12}
XE = np.array([0, 0.16, 2.02, 3.87, 5.86, 7.84, 9.82, 11.89, 13.96, 16.12, 18.32, 20.57])
XMID = 0.5 * (XE[:-1] + XE[1:])
XLAB = "depth inside the ECAL [$X_0$]"
LAYERS = (1, 2, 3, 4, 5, 6)
DESIGNS = [("no strips", ())] + [(f"strips only in layer {L}", (L,)) for L in LAYERS] + [("strips in layers 1-6", LAYERS)]
VARS = ["Eratio", "dE", "w_s3", "w_tot", "f_side", "E2/E"]
W0 = 4.0
METHODS = {"fit": "line fit, S-curve, mean layer weights (07)", "fit_event": "line fit, per-event layer weights",
           "vertex": "line fit through z = 0 (vertex constrained)", "log": f"log-weight barycentres (W0={W0}) + line fit",
           "pca_layer": "PCA of the S-corrected layer barycentres", "pca_hits": "PCA of all hits, S-corrected, energy weights",
           "pca_hits_log": f"PCA of all hits, S-corrected, log weights (W0={W0})"}


def style(strips):
    if not strips:
        return dict(color="k", ls=":")
    if len(strips) > 1:
        return dict(color="gray", ls="--")
    return dict(color=plt.cm.tab10(strips[0] - 1), ls="-", marker="o")


def ecolors(Es):
    return dict(zip(Es, plt.cm.viridis(np.linspace(0, 0.9, len(Es)))))


def h68(x):
    return 0.5 * np.diff(np.nanpercentile(x, [16, 84]))[0]


def save(fig, name, title):
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=100)
    plt.close(fig)


@lru_cache(None)
def load(conf, part, E):
    d = dict(np.load(f"{RED}/{conf}_{part}_{E}GeV.npz"))
    d["u"] = (d["theta"] - OFFSET) / GRID
    d["u0"] = np.round(d["u"]).astype(int)
    d["even"] = np.arange(len(d["u"])) % 2 == 0
    return d


def cells(q, u0):
    start = (u0 - HALF) % 4
    pad = np.zeros((len(q), 64))
    for s in range(4):
        pad[start == s, s:s + NB] = q[start == s]
    return pad.reshape(len(q), 16, 4).sum(2), 4 * np.arange(16) + 1.5 - HALF - start[:, None]


def readout(d, l, strip):
    q = d["prof"][:, l]
    if strip:
        return q, np.broadcast_to(np.arange(NB) - HALF, q.shape).astype(float), 1
    c, pos = cells(q, d["u0"])
    return c, pos, 4


def shape_vars(q, pos, step):
    n, m = q.shape
    imax = np.argmax(np.where(np.abs(pos) <= 12, q, -1e9), 1)
    E1 = q[np.arange(n), imax]
    d = np.abs(pos - pos[np.arange(n), imax][:, None]) / step
    w = lambda r: np.where(d <= r, q, 0)
    tot = w(16 / step).sum(1) + 1e-9
    loc = np.zeros_like(q, bool)
    loc[:, 1:-1] = (q[:, 1:-1] > q[:, :-2]) & (q[:, 1:-1] >= q[:, 2:]) & (q[:, 1:-1] > 0.01 * tot[:, None])
    loc &= (d > 0) & (d <= 16 / step)
    E2 = np.where(loc, q, 0).max(1)
    i2 = np.argmax(np.where(loc, q, -1), 1)
    j = np.arange(m)
    inside = (j > np.minimum(imax, i2)[:, None]) & (j < np.maximum(imax, i2)[:, None])
    valley = np.where(inside, q, np.inf).min(1)
    dE = np.where(E2 > 0, E2 - np.where(np.isfinite(valley), valley, E2), 0)
    bary = (w(16 / step) * pos).sum(1) / tot
    wtot = np.sqrt((w(16 / step) * (pos - bary[:, None]) ** 2).sum(1) / tot)
    c1, c3 = w(1.1).sum(1), w(3.1).sum(1)
    ws3 = np.sqrt((w(1.1) * d ** 2).sum(1) / (c1 + 1e-9))
    return np.stack([(E1 - E2) / (E1 + E2 + 1e-9), dE / tot, ws3, wtot, (c3 - c1) / (c1 + 1e-9), E2 / tot], 1)


def features(d, strips):
    f = [d["prof"].sum(2)[:, :7] / (d["prof"].sum((1, 2))[:, None] + 1e-9)]
    names = [f"E frac L{l}" for l in range(7)]
    for l in LAYERS:
        for strip in (False, True) if l in strips else (False,):
            f.append(shape_vars(*readout(d, l, strip)))
            names += [f"{v} L{l} {'strips' if strip else 'cells'}" for v in VARS]
    return np.concatenate(f, 1), names


@lru_cache(None)
def bdt(conf, E, strips):
    (Xg, names), (Xp, _) = features(load(conf, "gamma", E), strips), features(load(conf, "pi0", E), strips)
    X, y = np.concatenate([Xg, Xp]), np.r_[np.zeros(len(Xg)), np.ones(len(Xp))]
    half = np.random.default_rng(7).random(len(y)) < 0.5
    score, models = np.zeros(len(y)), []
    for train in (half, ~half):
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                           l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                           n_iter_no_change=20, random_state=1).fit(X[train], y[train])
        score[~train] = m.predict_proba(X[~train])[:, 1]
        models.append(m)
    cut = np.quantile(score[y == 0], 0.9)
    eff = max((score[y == 1] <= cut).mean(), 1 / (y == 1).sum())
    fpr, tpr, _ = roc_curve(1 - y, -score)
    return dict(rej=1 / eff, err=np.sqrt((1 - eff) / ((y == 1).sum() * eff)) / eff, tpr=tpr, fpr=fpr,
                score=score, y=y, cut=cut, X=X, names=names, model=models[0], test=~half)


def log_weights(q, total, w0):
    return np.where(q > 0, np.maximum(0, w0 + np.log(np.maximum(q, 1e-12) / total[:, None])), 0)


def barycentre(d, l, strip, w0=None):
    q, pos, step = readout(d, l, strip)
    c = pos[np.arange(len(q)), np.argmax(np.where(np.abs(pos) <= 12, q, -1e9), 1)]
    if w0 is None:
        w = np.where(np.abs(pos - c[:, None]) <= (5 if strip else 4.1), q, 0)
    else:
        q = np.where(np.abs(pos - c[:, None]) <= 16, q, 0)
        w = log_weights(q, q.sum(1), w0)
    ub = d["u0"] + np.where(w.sum(1) > 0, (w * pos).sum(1) / np.maximum(w.sum(1), 1e-12), np.nan)
    if w0 is not None:
        return ub, np.zeros(len(ub))
    res = ub - d["u"]
    b = (np.mod(np.nan_to_num(ub) + 0.5, step) / step * 20).astype(int).clip(0, 19)
    corr = np.array([np.nanmedian(res[d["even"] & (b == i)]) if (d["even"] & (b == i)).sum() > 5 else 0 for i in range(20)])
    return ub - corr[b], corr[b]


def dz(d, l, strip, w0=None):
    return -(barycentre(d, l, strip, w0)[0] - d["u"]) * GRID * d["rho"][l] / np.sin(d["theta"]) ** 2


def principal_axis(rho, z, w, axis):
    sw = w.sum(axis)
    rm, zm = (w * rho).sum(axis) / sw, (w * z).sum(axis) / sw
    rc, zc = rho - np.expand_dims(rm, axis), z - np.expand_dims(zm, axis)
    b = np.tan(0.5 * np.arctan2(2 * (w * rc * zc).sum(axis), (w * rc ** 2).sum(axis) - (w * zc ** 2).sum(axis)))
    return b, zm - b * rm


def pca_hits(d, strips, w0):
    rho, z, w = [], [], []
    for l in range(1, 11):
        q, pos, _ = readout(d, l, l in strips)
        shift = barycentre(d, l, l in strips)[1]
        rho.append(np.full(q.shape, d["rho"][l]))
        z.append(d["rho"][l] / np.tan(OFFSET + GRID * (d["u0"][:, None] + pos - shift[:, None])))
        w.append(q)
    rho, z, w = np.concatenate(rho, 1), np.concatenate(z, 1), np.concatenate(w, 1)
    if w0 is not None:
        w = log_weights(w, w.sum(1), w0)
    return principal_axis(rho, z, w, 1)


@lru_cache(None)
def pointing(conf, E, strips, method="fit", w0=W0):
    d = load(conf, "gamma", E)
    if method.startswith("pca_hits"):
        b, z0 = pca_hits(d, strips, w0 if method == "pca_hits_log" else None)
    else:
        rho, z, w = [], [], []
        for l in range(1, 11):
            El = d["prof"][:, l].sum(1)
            good = El > 0.01 * E
            if (good & d["even"]).sum() < 50:
                continue
            dzl = dz(d, l, l in strips, w0 if method == "log" else None)
            good &= np.isfinite(dzl)
            sigma = h68(dzl[good & d["even"]])
            if method in ("fit", "vertex"):
                w.append(good / sigma ** 2)
            elif method == "pca_layer":
                w.append(good * El)
            else:
                w.append(good * El / (sigma ** 2 * El[good & d["even"]].mean()))
            rho.append(np.full(len(El), d["rho"][l]))
            z.append(d["rho"][l] / np.tan(d["theta"]) + np.nan_to_num(dzl))
        rho, z, w = np.array(rho), np.array(z), np.array(w)
        if method == "vertex":
            b, z0 = (w * rho * z).sum(0) / (w * rho ** 2).sum(0), np.full(len(d["u"]), np.nan)
        elif method == "pca_layer":
            b, z0 = principal_axis(rho, z, w, 0)
        else:
            sw, sx, sy = w.sum(0), (w * rho).sum(0), (w * z).sum(0)
            b = (sw * (w * rho * z).sum(0) - sx * sy) / (sw * (w * rho ** 2).sum(0) - sx ** 2)
            z0 = (sy - b * sx) / sw
    odd = ~d["even"]
    return h68((np.arctan2(1, b) - d["theta"])[odd]) * 1e3, h68(z0[odd]) * 10


def gauss_core(x):
    mu, sg = np.median(x), 1.4826 * np.median(np.abs(x - np.median(x)))
    for _ in range(4):
        h, e = np.histogram(x, 60, (mu - 2 * sg, mu + 2 * sg))
        (_, mu, sg), cov = curve_fit(lambda t, a, m, s: a * np.exp(-0.5 * ((t - m) / s) ** 2), 0.5 * (e[1:] + e[:-1]), h,
                                     p0=[h.max(), mu, sg], sigma=np.sqrt(np.maximum(h, 1)))
        sg = abs(sg)
    return mu, sg, np.sqrt(cov[2, 2])


def sw_energy(conf, E):
    f = f"{RED}/{conf}_gamma_{E}GeV_swE.npy"
    if not os.path.exists(f):
        import uproot, awkward as ak
        b = "AugmentedEMBCaloClusters/AugmentedEMBCaloClusters.energy"
        a = uproot.open(f"{RECO[conf]}/stripseverywhere_reco_LArPb_10k_gamma_{E}GeV.root")["events"].arrays([b])[b]
        np.save(f, ak.to_numpy(ak.fill_none(ak.max(a, axis=1), 0.)))
    return np.load(f)


def plot_longitudinal():
    Es = [1, 5, 20, 50, 100]
    col = ecolors(Es)
    frac = {(c, E): load(c, "gamma", E)["prof"].sum(2).mean(0) / load(c, "gamma", E)["prof"].sum((1, 2)).mean()
            for c in CONFS for E in Es}
    fig, axs = plt.subplots(1, 3, figsize=(20, 5.5))
    for ax, conf in zip(axs, CONFS):
        for E in Es:
            ax.step(XE, np.r_[frac[conf, E], frac[conf, E][-1]] * 100, where="post", color=col[E], label=f"{E} GeV")
        for l in range(11):
            ax.text(XMID[l], 0.3, str(l), ha="center", fontsize=8, color="gray")
        ax.set(title=CONFS[conf], xlabel=XLAB + " (gray: layer)", ylabel="mean energy fraction in the layer [%]", ylim=(0, 26))
        ax.legend(title="$\\gamma$")
    for E in Es:
        axs[2].plot(range(1, 11), (frac["front", E] / frac["baseline", E])[1:], "o-", color=col[E], label=f"{E} GeV")
    axs[2].axhline(1, color="k", lw=0.8)
    axs[2].set(title="ratio front / behind, same layer", xlabel="layer", ylabel="energy fraction ratio", yscale="log", xticks=range(1, 11))
    axs[2].legend()
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
    save(fig, "01_longitudinal_gamma", "Longitudinal development of photon showers (sliding-window cluster cells)")


def plot_events():
    rows = [("gamma", 10), ("pi0", 10), ("gamma", 50), ("pi0", 50)]
    for conf in CONFS:
        fig, axs = plt.subplots(len(rows), len(DESIGNS), figsize=(4 * len(DESIGNS), 3.6 * len(rows)), sharey=True)
        for i, (part, E) in enumerate(rows):
            d = load(conf, part, E)
            ev = np.where(d["alpha"] > 0.004)[0][3] if part == "pi0" else 3
            vmax = np.log10(d["prof"][ev].max())
            for j, (name, strips) in enumerate(DESIGNS):
                ax = axs[i, j]
                for l in range(11):
                    q, pos, step = readout(d, l, l in strips)
                    edges = np.r_[pos[ev] - step / 2, pos[ev][-1] + step / 2]
                    ax.pcolormesh(edges * GRID * d["rho"][l], [l - 0.5, l + 0.5], np.log10(np.clip(q[ev], 1e-5, None))[None],
                                  vmin=-4, vmax=vmax, shading="flat")
                ax.set(xlim=(-10, 10), yticks=range(11))
                if i == 0:
                    ax.set_title(name)
                if i == len(rows) - 1:
                    ax.set_xlabel("z - z$_{axis}$ [cm]")
            axs[i, 0].set_ylabel(f"{part} {E} GeV\nlayer")
            if part == "pi0":
                axs[i, 0].text(-9.5, 10.3, f"E$_\\gamma$ = {d['egam'][ev][0]:.1f}, {d['egam'][ev][1]:.1f} GeV, "
                                           f"$\\alpha$ = {d['alpha'][ev] * 1e3:.0f} mrad", color="w", fontsize=8, va="top")
        save(fig, f"02_z_development_{conf}",
             f"Single events seen by each readout design (4-strip cells where there are no strips; colour = log10 E/GeV) - {CONFS[conf]}")

    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (part, E) in zip(axs, rows):
        m = {c: load(c, part, E)["prof"].mean(0) / load(c, part, E)["prof"].sum((1, 2)).mean() for c in CONFS}
        rho = load("baseline", part, E)["rho"]
        for l in range(1, 11):
            r = np.where(m["baseline"][l] > 1e-4, m["front"][l] / np.maximum(m["baseline"][l], 1e-12), np.nan)
            im = ax.pcolormesh((np.arange(NB + 1) - HALF - 0.5) * GRID * rho[l], [l - 0.5, l + 0.5], r[None], cmap="RdBu_r",
                               shading="flat", norm=matplotlib.colors.LogNorm(0.5, 2))
        ax.set(xlim=(-8, 8), yticks=range(1, 11), xlabel="z - z$_{axis}$ [cm]", ylabel="layer", title=f"{part} {E} GeV")
    fig.colorbar(im, ax=axs, label="energy fraction, front / behind", ticks=[0.5, 0.7, 1, 1.4, 2], format="%.1f")
    fig.suptitle("Ratio of the mean z development, solenoid in front / behind ECAL, same layer and strip (white: <1e-4 of E)")
    fig.savefig(f"{OUT}/02_z_development_ratio.png", dpi=100)
    plt.close(fig)


def plot_lateral():
    Es = [5, 20, 50, 100]
    col = ecolors(Es)
    W = {}
    for conf in CONFS:
        for E in Es:
            d = load(conf, "gamma", E)
            p = d["prof"]
            x = (np.arange(NB) - HALF - (d["u"] - d["u0"])[:, None])[:, None, :]
            w = np.where(np.abs(x) <= 10, p, 0)
            rms = np.sqrt((w * x ** 2).sum(2) / (w.sum(2) + 1e-12))
            core = np.where(np.abs(x) <= 1.5, p, 0).sum(2) / (p.sum(2) + 1e-12)
            has = p.sum(2) > 0.005 * E
            W[conf, E] = (np.array([np.median(rms[has[:, l], l]) * GRID * d["rho"][l] * 10 for l in range(1, 11)]),
                          np.array([np.median(core[has[:, l], l]) for l in range(1, 11)]))
    for xname, xv, xlab in [("X0", XMID[1:], XLAB), ("layer", np.arange(1, 11), "layer")]:
        fig, axs = plt.subplots(2, 3, figsize=(20, 9))
        for j, conf in enumerate(CONFS):
            for E in Es:
                axs[0, j].plot(xv, W[conf, E][0], "o-", color=col[E], label=f"{E} GeV")
                axs[1, j].plot(xv, W[conf, E][1], "o-", color=col[E])
            axs[0, j].axhline(GRID * 2230, color="k", ls=":")
            axs[0, j].set(title=CONFS[conf], ylabel="median lateral RMS in z [mm]", ylim=(0, 28))
            axs[1, j].set(xlabel=xlab, ylabel="median fraction of the layer energy\nin the 3 strips around the photon axis", ylim=(0.1, 1))
            axs[0, j].legend(title="$\\gamma$ (dotted: strip pitch)")
        for E in Es:
            axs[0, 2].plot(xv, W["front", E][0] / W["baseline", E][0], "o-", color=col[E])
            axs[1, 2].plot(xv, W["front", E][1] / W["baseline", E][1], "o-", color=col[E])
        axs[0, 2].set(title="ratio front / behind, same layer", ylabel="RMS ratio")
        axs[1, 2].set(xlabel=xlab, ylabel="core fraction ratio")
        for ax in axs.ravel():
            ax.grid(alpha=0.3)
        for ax in axs[:, 2]:
            ax.axhline(1, color="k", lw=0.8)
        save(fig, f"03_lateral_width_vs_{xname}", "Lateral (z) size of photon showers per layer at strip granularity "
             "(what a strip layer placed there would see); events with >0.5% of E in the layer")


def plot_lateral_designs():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for i, conf in enumerate(CONFS):
        for j, E in enumerate([20, 50]):
            d = load(conf, "gamma", E)
            for name, strips in DESIGNS:
                rms, sz = [], []
                for l in range(1, 11):
                    q, pos, _ = readout(d, l, l in strips)
                    c = pos[np.arange(len(q)), np.argmax(np.where(np.abs(pos) <= 12, q, -1e9), 1)]
                    w = np.where(np.abs(pos - c[:, None]) <= 16, q, 0)
                    tot = w.sum(1) + 1e-12
                    b = (w * pos).sum(1) / tot
                    ok = d["prof"][:, l].sum(1) > 0.005 * E
                    rms.append(np.median(np.sqrt((w * (pos - b[:, None]) ** 2).sum(1) / tot)[ok]) * GRID * d["rho"][l] * 10)
                    sz.append(h68(dz(d, l, l in strips)[ok & ~d["even"]]) * 10)
                axs[i, j].plot(range(1, 11), rms, label=name, **style(strips))
                axs2[i, j].plot(range(1, 11), sz, label=name, **style(strips))
            axs[i, j].set(title=f"{CONFS[conf]}, $\\gamma$ {E} GeV", ylabel="median measured RMS in z [mm]")
            axs2[i, j].set(title=f"{CONFS[conf]}, $\\gamma$ {E} GeV", ylabel="z resolution of the layer barycentre [mm]", yscale="log")
    for ax in list(axs.ravel()) + list(axs2.ravel()):
        ax.grid(alpha=0.3, which="both")
        ax.set_xticks(range(1, 11))
    for ax in list(axs[1]) + list(axs2[1]):
        ax.set_xlabel("layer")
    axs[0, 0].legend(fontsize=8)
    axs2[0, 0].legend(fontsize=8)
    save(fig, "03b_measured_width_per_design", "Lateral width measured by each design, around the reconstructed barycentre")
    save(fig2, "03c_layer_position_resolution_per_design", "z resolution of the S-corrected barycentre of each layer, per design")

    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for i, conf in enumerate(CONFS):
        for j, part in enumerate(["pi0", "gamma"]):
            for name, strips in DESIGNS:
                frac = []
                for E in ENERGIES:
                    d = load(conf, part, E)
                    two = [shape_vars(*readout(d, l, l in strips))[:, 0] < 0.9 / 1.1 for l in LAYERS]
                    frac.append(np.any(two, 0).mean())
                axs[i, j].plot(ENERGIES, frac, label=name, **style(strips))
            what = "$\\pi^0$ (wanted)" if part == "pi0" else "$\\gamma$ (fakes)"
            axs[i, j].set(title=f"{CONFS[conf]}: {what}", xscale="log", ylabel="fraction of events with two peaks", ylim=(0, 1))
            axs[i, j].grid(alpha=0.3, which="both")
    for ax in axs[1]:
        ax.set_xlabel("E [GeV]")
    axs[0, 0].legend(fontsize=8)
    save(fig, "03d_two_peaks_per_design", "Two maxima seen in layers 1-6 (second local maximum > 10% of the first, within 16 strips)")


def plot_log_checks():
    shown = [DESIGNS[0], DESIGNS[3], DESIGNS[-1]]
    for conf in CONFS:
        fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        for j, E in enumerate([20, 50]):
            d = load(conf, "gamma", E)
            for name, strips in shown:
                for w0, ls in [(None, "-"), (W0, "--")]:
                    rms, sz = [], []
                    for l in range(1, 11):
                        q, pos, _ = readout(d, l, l in strips)
                        c = pos[np.arange(len(q)), np.argmax(np.where(np.abs(pos) <= 12, q, -1e9), 1)]
                        q = np.where(np.abs(pos - c[:, None]) <= 16, q, 0)
                        w = q if w0 is None else log_weights(q, q.sum(1), w0)
                        sw = np.maximum(w.sum(1), 1e-12)
                        b = (w * pos).sum(1) / sw
                        ok = d["prof"][:, l].sum(1) > 0.005 * E
                        rms.append(np.median(np.sqrt((w * (pos - b[:, None]) ** 2).sum(1) / sw)[ok]) * GRID * d["rho"][l] * 10)
                        sz.append(h68(dz(d, l, l in strips, w0)[ok & ~d["even"]]) * 10)
                    st = dict(style(strips), ls=ls)
                    lab = f"{name}, {'linear (+S-curve)' if w0 is None else f'log, W0={W0}'}"
                    axs[0, j].plot(range(1, 11), rms, label=lab, **st)
                    axs[1, j].plot(range(1, 11), sz, label=lab, **st)
            axs[0, j].set(title=f"$\\gamma$ {E} GeV", ylabel="median RMS in z with these weights [mm]")
            axs[1, j].set(xlabel="layer", ylabel="z resolution of the layer barycentre [mm]", yscale="log", xticks=range(1, 11))
        axs[0, 0].legend(fontsize=8)
        for ax in axs.ravel():
            ax.grid(alpha=0.3, which="both")
        save(fig, f"03e_log_vs_linear_weights_{conf}", "Shower shape with linear weights (solid; position S-corrected) "
             f"and log weights (dashed, W0={W0}) - {CONFS[conf]}")


def plot_unconverted():
    Es = [1, 5, 20, 50, 100]
    col = ecolors(Es)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, conf in zip(axs, CONFS):
        for E in Es:
            before = np.cumsum(load(conf, "gamma", E)["prof"].sum(2), 1) / E
            ax.plot(XE[1:7], [np.mean(before[:, L - 1] < 0.01) for L in LAYERS], "o-", color=col[E], label=f"{E} GeV")
        t = np.linspace(0, 12, 50)
        ax.plot(t, np.exp(-7 / 9 * (t + UPSTREAM[conf])), "k:", label="exp(-7/9 t), t from the IP")
        for L in LAYERS:
            ax.text(XE[L], 1.3, f"L{L}", ha="center", fontsize=8, color="gray")
        ax.set(title=CONFS[conf], xlabel="depth of the front of the layer inside the ECAL [$X_0$]", yscale="log", ylim=(1e-3, 2))
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axs[0].set_ylabel("fraction of photons with < 1% of E deposited before the layer")
    save(fig, "04_unconverted_photons", "Photons still unconverted when they reach layer L")


def plot_pid():
    fig, axs = plt.subplots(1, 3, figsize=(21, 5.5))
    for ax, conf in zip(axs, CONFS):
        for name, strips in DESIGNS:
            r = [bdt(conf, E, strips) for E in ENERGIES]
            ax.errorbar(ENERGIES, [x["rej"] for x in r], [x["err"] for x in r], capsize=2, label=name, **style(strips))
            print(conf, name, np.round([x["rej"] for x in r], 1), flush=True)
        ax.set(xscale="log", yscale="log", xlabel="E [GeV]", ylabel="$\\pi^0$ rejection at 90% $\\gamma$ efficiency",
               title=CONFS[conf], ylim=(1, 20))
        ax.legend(fontsize=8)
    for name, strips in DESIGNS[1:-1]:
        ratio = [bdt("front", E, strips)["rej"] / bdt("baseline", E, strips)["rej"] for E in ENERGIES]
        axs[2].plot(ENERGIES, ratio, label=name, **style(strips))
    axs[2].axhline(1, color="k", lw=0.8)
    axs[2].set(xscale="log", xlabel="E [GeV]", ylabel="rejection ratio front / behind", title="ratio, same design")
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
    save(fig, "05_pi0_rejection", "$\\gamma/\\pi^0$ BDT (2-fold cross-validated, no noise). "
         "At 1-2 GeV the two photons are mostly outside the cluster window")

    shown = [DESIGNS[i] for i in (0, 1, 2, 3, 4, -1)]
    for conf in CONFS:
        fig, axs = plt.subplots(1, 4, figsize=(22, 5.5), sharey=True)
        for ax, E in zip(axs, [10, 20, 50, 100]):
            for name, strips in DESIGNS:
                r = bdt(conf, E, strips)
                ax.plot(r["tpr"], 1 / np.maximum(r["fpr"], 1e-4), label=name, **{k: v for k, v in style(strips).items() if k != "marker"})
            ax.axvline(0.9, color="gray", lw=0.8)
            ax.set(title=f"{E} GeV", xlabel="$\\gamma$ efficiency", xlim=(0.5, 1), yscale="log", ylim=(1, 100))
            ax.grid(alpha=0.3, which="both")
        axs[0].set_ylabel("$\\pi^0$ rejection")
        axs[0].legend(fontsize=8)
        save(fig, f"06_roc_{conf}", f"ROC curves of the $\\gamma/\\pi^0$ BDT - {CONFS[conf]}")

        Es = [10, 20, 50, 100]
        fig, axs = plt.subplots(len(shown), len(Es), figsize=(5 * len(Es), 2.8 * len(shown)), sharex=True)
        for i, (name, strips) in enumerate(shown):
            for j, E in enumerate(Es):
                r = bdt(conf, E, strips)
                for k, lab in enumerate(["$\\gamma$", "$\\pi^0$"]):
                    axs[i, j].hist(r["score"][r["y"] == k], np.linspace(0, 1, 51), histtype="step", density=True, label=lab)
                axs[i, j].axvline(r["cut"], color="k", ls=":")
                axs[i, j].set(yscale="log", title=f"{name}, {E} GeV")
        axs[0, 0].legend()
        save(fig, f"06b_bdt_score_{conf}", f"BDT score, dotted: cut at 90% $\\gamma$ efficiency - {CONFS[conf]}")

        for E in (20, 50):
            g, p = load(conf, "gamma", E), load(conf, "pi0", E)
            fig, axs = plt.subplots(7, 6, figsize=(24, 22))
            for i, L in enumerate(LAYERS):
                vg, vp = shape_vars(*readout(g, L, True)), shape_vars(*readout(p, L, True))
                for k in range(6):
                    b = np.linspace(*np.percentile(np.r_[vg[:, k], vp[:, k]], [0.5, 99.5]) + [0, 1e-9], 50)
                    axs[i, k].hist(vg[:, k], b, histtype="step", density=True, label="$\\gamma$")
                    axs[i, k].hist(vp[:, k], b, histtype="step", density=True, label="$\\pi^0$")
                    axs[i, k].set_title(f"{VARS[k]}, strips in layer {L}", fontsize=9)
            fg, fp = (x["prof"].sum(2) / (x["prof"].sum((1, 2))[:, None] + 1e-9) for x in (g, p))
            for k in range(6):
                b = np.linspace(0, max(np.percentile(fg[:, k], 99.5), 1e-3), 50)
                axs[6, k].hist(fg[:, k], b, histtype="step", density=True, label="$\\gamma$")
                axs[6, k].hist(fp[:, k], b, histtype="step", density=True, label="$\\pi^0$")
                axs[6, k].set_title(f"energy fraction in layer {k}", fontsize=9)
            axs[0, 0].legend()
            save(fig, f"06c_bdt_inputs_{conf}_{E}GeV", f"BDT inputs, $\\gamma$ vs $\\pi^0$ at {E} GeV - {CONFS[conf]}\n"
                 "Eratio=(E1-E2)/(E1+E2); dE=valley depth/E; w_s3=width in +-1 channel; w_tot=width in +-16 strips; "
                 "f_side=(E+-3-E+-1)/E+-1; E2/E")

        fig, axs = plt.subplots(2, len(shown), figsize=(5 * len(shown), 10))
        for i, E in enumerate((20, 50)):
            for j, (name, strips) in enumerate(shown):
                r = bdt(conf, E, strips)
                imp = permutation_importance(r["model"], r["X"][r["test"]], r["y"][r["test"]], scoring="roc_auc",
                                             n_repeats=3, random_state=1)
                top = np.argsort(imp.importances_mean)[::-1][:12]
                axs[i, j].barh(range(12)[::-1], imp.importances_mean[top], xerr=imp.importances_std[top])
                axs[i, j].set_yticks(range(12)[::-1], [r["names"][k] for k in top], fontsize=8)
                axs[i, j].set(title=f"{name}, {E} GeV", xlabel="AUC loss when shuffled")
        save(fig, f"06d_bdt_importance_{conf}", f"BDT permutation importance (12 most important inputs) - {CONFS[conf]}")


def plot_pointing():
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for j, conf in enumerate(CONFS):
        for name, strips in DESIGNS:
            r = np.array([pointing(conf, E, strips) for E in ENERGIES])
            for i in (0, 1):
                axs[i, j].plot(ENERGIES, r[:, i], label=name, **style(strips))
            print(conf, "pointing", name, np.round(r[:, 0], 1), flush=True)
        axs[0, j].set(title=CONFS[conf], ylabel="$\\sigma_\\theta$: 68% half-width of $\\theta_{reco}-\\theta_{true}$ [mrad]", yscale="log")
        axs[1, j].set(xlabel="E$_\\gamma$ [GeV]", ylabel="$\\sigma_{z_0}$: 68% half-width of z at the beam line [mm]",
                      xscale="log", yscale="log")
        for ax in axs[:, j]:
            ax.grid(alpha=0.3, which="both")
    axs[0, 0].legend(fontsize=8)
    save(fig, "07_pointing", "Photon pointing: weighted line fit to the S-corrected barycentres of layers 1-10, "
         "each read with the granularity of the design (no noise)")


def plot_pointing_methods():
    for conf in CONFS:
        fig, axs = plt.subplots(2, len(METHODS), figsize=(5 * len(METHODS), 10), sharex=True, sharey="row")
        for j, (method, title) in enumerate(METHODS.items()):
            for name, strips in DESIGNS:
                r = np.array([pointing(conf, E, strips, method) for E in ENERGIES])
                axs[0, j].plot(ENERGIES, r[:, 0], label=name, **style(strips))
                axs[1, j].plot(ENERGIES, r[:, 1], label=name, **style(strips))
                print(conf, method, name, np.round(r[:, 0], 1), flush=True)
            axs[0, j].set(title=title, yscale="log")
            axs[1, j].set(xlabel="E$_\\gamma$ [GeV]", xscale="log", yscale="log")
        axs[0, 0].set_ylabel("$\\sigma_\\theta$ [mrad]")
        axs[1, 0].set_ylabel("$\\sigma_{z_0}$ [mm] (not defined with the vertex constraint)")
        axs[0, 0].legend(fontsize=8)
        for ax in axs.ravel():
            ax.grid(alpha=0.3, which="both")
        save(fig, f"07b_pointing_methods_{conf}", f"Photon pointing with different methods, per design - {CONFS[conf]}")

        shown = [DESIGNS[i] for i in (0, 1, 2, 3, -1)]
        fig, axs = plt.subplots(2, len(shown), figsize=(5 * len(shown), 9), sharex=True, sharey="row")
        for j, (name, strips) in enumerate(shown):
            ref = np.array([pointing(conf, E, strips)[0] for E in ENERGIES])
            for k, (method, title) in enumerate(METHODS.items()):
                r = np.array([pointing(conf, E, strips, method)[0] for E in ENERGIES])
                axs[0, j].plot(ENERGIES, r, "o-", color=f"C{k}", label=title)
                axs[1, j].plot(ENERGIES, r / ref, "o-", color=f"C{k}")
            axs[0, j].set(title=name, yscale="log")
            axs[1, j].set(xlabel="E$_\\gamma$ [GeV]", xscale="log")
            axs[1, j].axhline(1, color="k", lw=0.8)
        axs[0, 0].set_ylabel("$\\sigma_\\theta$ [mrad]")
        axs[1, 0].set_ylabel("$\\sigma_\\theta$ / $\\sigma_\\theta$ (line fit, 07)")
        axs[0, 0].legend(fontsize=8)
        for ax in axs.ravel():
            ax.grid(alpha=0.3, which="both")
        save(fig, f"07c_pointing_method_comparison_{conf}", f"Pointing methods compared, design by design - {CONFS[conf]}")

    w0s = [2, 3, 4, 5, 6, 7]
    shown = [DESIGNS[0], DESIGNS[1], DESIGNS[3], DESIGNS[-1]]
    fig, axs = plt.subplots(2, 4, figsize=(24, 10), sharex=True)
    for i, conf in enumerate(CONFS):
        for j, (method, E) in enumerate([("log", 10), ("log", 50), ("pca_hits_log", 10), ("pca_hits_log", 50)]):
            for name, strips in shown:
                axs[i, j].plot(w0s, [pointing(conf, E, strips, method, w0)[0] for w0 in w0s], label=name, **style(strips))
            axs[i, j].set(title=f"{CONFS[conf]}\n{'log-weight barycentres + fit' if method == 'log' else 'PCA of S-corrected hits, log weights'}, "
                                f"$\\gamma$ {E} GeV", ylabel="$\\sigma_\\theta$ [mrad]")
            axs[i, j].grid(alpha=0.3)
    for ax in axs[1]:
        ax.set_xlabel("W0")
    axs[0, 0].legend(fontsize=8)
    save(fig, "07d_pointing_w0_scan", "Scan of the log-weight parameter W0 (cells below exp(-W0) of the energy get zero weight)")


def plot_energy():
    fig, axs = plt.subplots(1, 4, figsize=(26, 5.5))
    res = {}
    for conf in CONFS:
        res[conf] = []
        for E in ENERGIES:
            e = sw_energy(conf, E)
            mu, sg, err = gauss_core(e[e > 0.3 * E] / E)
            res[conf].append((mu, sg / mu, err / mu))
        res[conf] = np.array(res[conf])
        (a, c), _ = curve_fit(lambda E, a, c: np.sqrt(a ** 2 / E + c ** 2), ENERGIES, res[conf][:, 1], sigma=res[conf][:, 2], p0=[0.08, 0.005])
        a, c = abs(a), abs(c)
        line = axs[0].plot(ENERGIES, res[conf][:, 0], "o-", label=CONFS[conf])[0]
        axs[1].errorbar(ENERGIES, res[conf][:, 1] * 100, res[conf][:, 2] * 100, fmt="o", color=line.get_color(),
                        label=f"{CONFS[conf]}: {a * 100:.1f}%/$\\sqrt{{E}}$ $\\oplus$ {c * 100:.2f}%")
        t = np.logspace(0, 2, 50)
        axs[1].plot(t, np.sqrt(a ** 2 / t + c ** 2) * 100, color=line.get_color())
        print(conf, "sigma/E [%]", np.round(res[conf][:, 1] * 100, 2), flush=True)
    axs[2].plot(ENERGIES, res["front"][:, 1] / res["baseline"][:, 1], "o-")
    axs[2].axhline(1, color="k", lw=0.8)

    for conf, ls in [("baseline", "-"), ("front", "--")]:
        sigma = {}
        for name, strips in DESIGNS[:-1]:
            sigma[name] = []
            for E in ENERGIES:
                d = load(conf, "gamma", E)
                c, pos = cells(d["prof"].sum(1), d["u0"])
                lo = pos[np.arange(len(c)), np.argmax(c, 1)] - 18
                tot = sum(np.where((p >= lo[:, None]) & (p < lo[:, None] + 36), q, 0).sum(1)
                          for q, p, _ in (readout(d, l, l in strips) for l in range(11)))
                sigma[name].append(gauss_core(tot / E)[1])
            if strips:
                axs[3].plot(ENERGIES, np.array(sigma[name]) / np.array(sigma["no strips"]), marker="o", ls=ls,
                            color=style(strips)["color"], label=name if conf == "baseline" else None)
    axs[0].set(xscale="log", xlabel="E$_\\gamma$ [GeV]", ylabel="E$_{SW}$/E (Gaussian core)", title="response (no upstream correction)")
    axs[1].set(xscale="log", yscale="log", xlabel="E$_\\gamma$ [GeV]", ylabel="$\\sigma_E$/E [%]", title="resolution")
    axs[2].set(xscale="log", xlabel="E$_\\gamma$ [GeV]", ylabel="$\\sigma_E$/E front / behind", title="ratio")
    axs[3].set(xscale="log", xlabel="E$_\\gamma$ [GeV]", ylabel="$\\sigma_E$/E (design) / $\\sigma_E$/E (no strips)",
               title="designs, 9-tower window (solid: behind, dashed: front)", ylim=(0.99, 1.01))
    for ax in axs:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    save(fig, "08_energy_resolution", "Photon energy, leading sliding-window cluster. Without noise the cluster energy "
         "does not depend on where the strips are")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    plot_longitudinal()
    plot_events()
    plot_lateral()
    plot_lateral_designs()
    plot_log_checks()
    plot_unconverted()
    plot_pid()
    plot_pointing()
    plot_pointing_methods()
    plot_energy()
    print("done", OUT)
