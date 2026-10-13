import sys, math, os
import numpy as np, uproot, awkward as ak

DIRS = {"baseline": "/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output/baseline",
        "front": "/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output/solenoid_front_studies"}
OUT = "/eos/user/m/mcampana/ALLEGRO/analysis/strip_layer/reduced"
GRID, OFFSET, HALF = 0.009817477 / 4, 0.5902785, 30
conf, part, E = sys.argv[1], sys.argv[2], sys.argv[3]
nev = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
f = f"{DIRS[conf]}/stripseverywhere_reco_LArPb_10k_{part}_{E}GeV.root"
C, M = "EMBCaloClusterCells", "MCParticles"
br = [f"{C}/{C}.cellID", f"{C}/{C}.energy", f"{C}/{C}.position.x", f"{C}/{C}.position.y"] + \
     [f"{M}/{M}.{v}" for v in ("PDG", "momentum.x", "momentum.y", "momentum.z", "vertex.x", "vertex.y", "vertex.z")]
NB = 2 * HALF + 1
prof, frac, truth, rho_sum, rho_w = [], [], [], np.zeros(11), np.zeros(11)
t = uproot.open(f)["events"]
for a in t.iterate(br, entry_stop=nev, step_size=500, library="ak"):
    for ev in range(len(a)):
        pdg = np.asarray(a[br[4]][ev]); px, py, pz = (np.asarray(a[b][ev]) for b in br[5:8])
        vr = np.sqrt(sum(np.asarray(a[b][ev]) ** 2 for b in br[8:11]))
        th0 = math.atan2(math.hypot(px[0], py[0]), pz[0]); ph0 = math.atan2(py[0], px[0])
        u = (th0 - OFFSET) / GRID; u0 = round(u)
        tr = [th0, ph0, math.sqrt(px[0] ** 2 + py[0] ** 2 + pz[0] ** 2), 0, 0, 0]
        if pdg[0] == 111:
            g = np.where((pdg == 22) & (vr < 1.))[0][:2]
            if len(g) == 2:
                p1, p2 = (np.array([px[i], py[i], pz[i]]) for i in g)
                e1, e2 = np.linalg.norm(p1), np.linalg.norm(p2)
                tr[3:] = [min(e1, e2), max(e1, e2), math.acos(np.clip(p1 @ p2 / e1 / e2, -1, 1))]
        cid = np.asarray(a[br[0]][ev]).astype(np.uint64); en = np.asarray(a[br[1]][ev])
        cid, iu = np.unique(cid, return_index=True); en = en[iu]
        x, y = np.asarray(a[br[2]][ev])[iu], np.asarray(a[br[3]][ev])[iu]
        lay = ((cid >> np.uint64(11)) & np.uint64(0xFF)).astype(int); th = ((cid >> np.uint64(30)) & np.uint64(0x3FF)).astype(int)
        p = np.zeros((11, NB))
        for l in range(11):
            m = lay == l
            if not m.any(): continue
            rho_sum[l] += (np.hypot(x[m], y[m]) * en[m]).sum(); rho_w[l] += en[m].sum()
            if l == 0:
                k = (th[m][:, None] + np.arange(4)[None] - u0 + HALF).ravel(); w = np.repeat(en[m] / 4, 4)
            else:
                k = th[m] - u0 + HALF; w = en[m]
            ok = (k >= 0) & (k < NB)
            p[l] = np.bincount(k[ok], weights=w[ok], minlength=NB)
        prof.append(p.astype(np.float32)); frac.append(u - u0); truth.append(tr)
os.makedirs(OUT, exist_ok=True)
np.savez_compressed(f"{OUT}/{conf}_{part}_{E}GeV.npz", prof=np.array(prof), ufrac=np.array(frac), truth=np.array(truth),
                    rho=rho_sum / np.maximum(rho_w, 1e-12) / 10.)
print(conf, part, E, len(prof))
