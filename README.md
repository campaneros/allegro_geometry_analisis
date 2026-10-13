# allegro_geometry_analisis

Analysis code for the readout-design studies of the ALLEGRO ECAL barrel (ALLEGRO_o1_v03). Right now it contains
one study: in which layer it pays most to have the fine theta strips, with the solenoid behind the ECAL and with
the solenoid in front of it.

The idea behind the designs is simple: the PCB, the lead and the LAr stay as they are, only the way the pads are
grouped into readout channels changes. Moving the strip layer does not change the channel count: a strip layer
costs 4 channels per tower wherever it is.

The simulation and reconstruction scripts will live in a separate repository.

## What is in here

```
analysis/strip_layer/
    reduce.py    reco files -> per-event theta profiles at strip granularity + MC truth
    analyse.py   builds the designs from the reduced files and makes all the plots
```

### Inputs

M. Koppitz's "stripseverywhere" reco samples on the DRD6 EOS area: gamma and pi0 at 1-100 GeV, 10k events each,
theta and phi flat in 89-91 deg, strips in layers 1-10, modules merged by two, no noise. Two configurations:

- solenoid behind the ECAL: `/eos/experiment/drd/drd6/allegro/noble-liquid-cal-rnd/simulation_output/baseline`
- solenoid in front of the ECAL (+0.81 X0 of aluminium): `.../simulation_output/solenoid_front_studies`

Those directories are only read. `reduce.py` uses the cells of the sliding-window clusters
(`EMBCaloClusterCells`), the `MCParticles`, and nothing else from the reconstruction.

### How a design is built

The layer that has strips keeps them; every other layer is merged into 4-strip cells aligned to the absolute theta
grid, which is what `mergedCells_Theta="4"` does in the official segmentation. Without noise the energy is
additive, so the merged cell holds exactly the energy a real 4-strip cell would have collected. The sliding window
works on towers that are 4 strips wide, so the clusters themselves are the same in every design. I checked that
the cluster edges fall on 4-strip boundaries in about 96% of the events; the rest are clusters whose outermost
strips are empty.

Designs compared:

- no strips (all layers as 4-strip cells);
- strips only in layer L, L = 1..6, cells everywhere else;
- strips in layers 1-6, cells in 7-10.

### What analyse.py produces

- longitudinal profile of photon showers per layer and in X0 (counted from the front of the ECAL, so upstream
  material is not included), plus the ratio between the two solenoid positions;
- single gamma and pi0 events as seen by each design, and the ratio of the mean z development;
- lateral width per layer: true shape at strip granularity (vs X0 and vs layer), the width each design actually
  measures, the z resolution of each layer's barycentre, and how often a design sees two separate peaks;
- the same shower-shape checks with log weights instead of linear ones. With strips narrower than the shower core
  the linear barycentre has little S-curve, so log weights mostly give more weight to the tails; with 4-strip cells
  they are an alternative to the S-curve correction, not something to add on top of it;
- fraction of photons that reach a layer before they start to shower;
- gamma/pi0 separation: a BDT on lateral-shape variables of layers 1-6 (Eratio, valley depth, widths, f_side, E2/E,
  computed on the 4-strip cells and, where the design has them, on the strips) plus the longitudinal fractions of
  layers 0-6. 2-fold cross-validated. Plots: pi0 rejection at 90% photon efficiency, ROC curves, score
  distributions, input variables, permutation importance;
- photon pointing, with several methods so they can be compared:
  - line fit (the main one, as ATLAS does it): energy-weighted barycentre of each layer 1-10, read with the
    granularity of the design and corrected for the S-curve, then a weighted straight-line fit z = z0 + rho*cot(theta)
    with weight 1/sigma_l^2 per layer (sigma_l averaged over the events); gives sigma_theta and sigma_z0;
  - the same fit with per-event layer weights (the error of a layer scaled as 1/sqrt(E_layer) of that event);
  - the same fit forced through z = 0 (vertex known): sigma_theta then becomes position resolution / radius,
    well below 1 mrad, a different scale from the free fit;
  - log-weighted barycentres (W0 = 4, the usual value from Awes et al.; ATLAS itself uses linear weights plus the
    S-curve correction) followed by the line fit;
  - principal axis (PCA), always after the S-curve correction, in two ways: on the corrected barycentre of each
    layer weighted with the layer energy, and on all the hits of the cluster, each shifted by the S-curve correction
    of its layer in that event, with energy or log weights. Unlike the line fit, the PCA minimises the perpendicular
    distance of the points from the axis;
  - a scan of W0 between 2 and 7 for the log-weighted methods.
  The radial position of a cell is the centre of its layer, so the depth of the shower inside a layer cannot be
  used event by event;
- energy: response and resolution of the leading sliding-window cluster for the two configurations, and the check
  that the resolution does not depend on where the strips are.

## How to run

You need the key4hep stack (I use the 2026-04-08 release) for uproot, awkward, scikit-learn and matplotlib, and
read access to the DRD6 EOS area.

```bash
source /cvmfs/sw.hsf.org/key4hep/setup.sh -r 2026-04-08
cd analysis/strip_layer

# 1. reduce: one call per sample (configuration, particle, energy, number of events)
for c in baseline front; do for p in gamma pi0; do for e in 1 2 5 10 20 35 50 80 100; do
  echo $c $p $e
done; done; done | xargs -P 8 -L 1 python3 reduce.py

# 2. plots
OMP_NUM_THREADS=2 python3 analyse.py
```

Reduced files go to `/eos/user/m/mcampana/ALLEGRO/analysis/strip_layer/reduced` (about 170 MB for 5000 events per
sample) and the plots to `/eos/user/m/mcampana/www/ALLEGRO/strip_layer/plots`. Change `OUT` and `RED` at the top
of the scripts if you run it as someone else.

Keep `OMP_NUM_THREADS` low. On a shared lxplus node scikit-learn otherwise takes every core and the BDT training
becomes hundreds of times slower.

## Things to keep in mind

- No noise and no cross-talk anywhere. The noise maps of the strip samples are not reliable, so they are left out;
  as a consequence the energy resolution cannot depend on where the strips are.
- The X0 per layer come from navigating the TGeo geometry at theta = 90 deg, averaged over the absorber period.
  Showers away from 90 deg see more material per layer.
- The S-curve corrections and the per-layer weights of the pointing fit are taken from MC truth on the even events
  and applied to the odd ones.
- 5000 events per sample. Differences of 10-15% between neighbouring strip layers are at the edge of what this can
  resolve.
- At 1-2 GeV the two photons of a pi0 are usually further apart than the cluster window, so the shape BDT is not the
  right tool there.
