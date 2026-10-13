import os
import sys
import numpy as np
import uproot

BASE = "/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output"
DIRS = {"baseline": f"{BASE}/baseline", "front": f"{BASE}/solenoid_front_studies"}
OUT = "/eos/user/m/mcampana/ALLEGRO/analysis/strip_layer/reduced"
GRID, OFFSET, HALF = 0.009817477 / 4, 0.5902785, 30
NB = 2 * HALF + 1

conf, part, E = sys.argv[1:4]
nev = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
C, M = "EMBCaloClusterCells/EMBCaloClusterCells.", "MCParticles/MCParticles."
names = [C + v for v in ("cellID", "energy", "position.x", "position.y")] + \
        [M + v for v in ("PDG", "momentum.x", "momentum.y", "momentum.z", "vertex.x", "vertex.y", "vertex.z")]
a = uproot.open(f"{DIRS[conf]}/stripseverywhere_reco_LArPb_10k_{part}_{E}GeV.root")["events"].arrays(
    names, entry_stop=nev, library="np")

os.makedirs(OUT, exist_ok=True)
n = len(a[names[0]])
prof = np.zeros((n, 11, NB), np.float32)
theta, alpha, egam = np.zeros(n), np.zeros(n), np.zeros((n, 2))
rho_sum, e_sum = np.zeros(11), np.zeros(11)
for i in range(n):
    pdg = a[M + "PDG"][i]
    p = np.stack([a[M + "momentum." + c][i] for c in "xyz"], 1)
    theta[i] = np.arctan2(np.hypot(p[0, 0], p[0, 1]), p[0, 2])
    if pdg[0] == 111:
        r = np.sqrt(sum(a[M + "vertex." + c][i] ** 2 for c in "xyz"))
        g = np.where((pdg == 22) & (r < 1))[0][:2]
        if len(g) == 2:
            e = np.linalg.norm(p[g], axis=1)
            egam[i] = np.sort(e)
            alpha[i] = np.arccos(np.clip(p[g[0]] @ p[g[1]] / e[0] / e[1], -1, 1))

    cid, first = np.unique(a[C + "cellID"][i].astype(np.uint64), return_index=True)
    en = a[C + "energy"][i][first]
    rho = np.hypot(a[C + "position.x"][i][first], a[C + "position.y"][i][first]) / 10
    lay = ((cid >> np.uint64(11)) & np.uint64(0xFF)).astype(int)
    k = ((cid >> np.uint64(30)) & np.uint64(0x3FF)).astype(int) - round((theta[i] - OFFSET) / GRID) + HALF
    np.add.at(rho_sum, lay, rho * en)
    np.add.at(e_sum, lay, en)

    pre = lay == 0
    lay = np.r_[lay[~pre], np.zeros(4 * pre.sum(), int)]
    k = np.r_[k[~pre], (k[pre, None] + np.arange(4)).ravel()]
    en = np.r_[en[~pre], np.repeat(en[pre] / 4, 4)]
    ok = (k >= 0) & (k < NB)
    np.add.at(prof[i], (lay[ok], k[ok]), en[ok])

np.savez_compressed(f"{OUT}/{conf}_{part}_{E}GeV.npz", prof=prof, theta=theta, alpha=alpha, egam=egam,
                    rho=rho_sum / np.maximum(e_sum, 1e-12))
print(conf, part, E, n)
