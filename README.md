# Badminton Shuttlecock Simulator (v9.2)

A physics-based badminton shuttlecock trajectory simulator — 2D (point-mass and
coupled translation+rotation) and 3D (6-DOF rigid body). Every coefficient and
constant in the aerodynamic model is either measured in the cited literature or
derived from those measurements; nothing is a free-fit parameter chosen to make
a particular trajectory look right. This repository exists so that a link in a
research report's citation can be opened and checked, not just read about.

## What it predicts, and how it's checked

- **Physics core**: `shuttlecock_simulator_9.2.py` — the full model, a Gradio web
  UI, and two automated verification suites.
- **No-server front end**: `shuttlecock_ipywidgets.py` — the same physics
  (imported by path, not copied) behind an `ipywidgets` UI that runs entirely
  inside a Jupyter/Colab kernel, with no HTTP server and no share link to fail.

The model is validated on three levels, all runnable from the file itself:

1. **Analytic limits** — zero-drag flight reduces to the textbook parabola
   (0.01% of the closed-form result), free fall converges to the theoretical
   terminal velocity `sqrt(2mg / rho·S·Cd)` (0.00% error), a damped/undamped/
   critically-damped oscillator matches its closed-form solution, energy is
   strictly non-increasing under drag alone, and angular momentum is exactly
   conserved when the net moment is zero.
2. **Numerical convergence** — Richardson extrapolation as `dt` is halved, and
   agreement between the RK4 and Euler integrators.
3. **Literature agreement** — aerodynamic length `ℓ = 2M/(ρ·S·Cd) ≈ 4.1 m`
   against Cohen et al.'s measured 4.04 m; terminal velocity ≈ 6.35 m/s against
   the ~25 km/h reported; range doubling by ~7.5% between 10 °C and 40 °C
   against the ~10% reported; and range growing only logarithmically with
   launch speed (the "aerodynamic wall"), so no launch condition sends a
   shuttlecock meaningfully past the 13.4 m court.

Running `ValidationEngine().run_all()` currently passes **21/21**, and
`physics_verification_suite()` currently passes **71/71**. Both are plain
functions in the module — anyone can import the file and re-run them.

## Physical basis (with sources)

| Component | Model | Source |
|---|---|---|
| Turnover (flip) rotation | Aerodynamic pendulum `φ̈ + β(U)φ̇ + ω₀(U)²sin φ = 0` | Cohen, Darbois Texier, Quéré, Clanet, *The physics of badminton*, New J. Phys. **17** 063001 (2015) |
| Aerodynamic length | `ℓ = 2M/(ρ·S·C_D)` | Cohen et al. 2015 |
| Attitude-dependent C_D, C_L | Cross-flow decomposition of a body of revolution: `C_D = Cd_ref + (Cd_cross−Cd_ref)sin²φ`, `C_L = ½(Cd_cross−Cd_ref)sin2φ` | Derived from the two measured anchors in Cohen et al. 2015 fig. 8b; shape matches Chan & Rossmann (2012) |
| Natural axial spin | `R·Ω/U ≈ 0.04` (feather) | Cohen et al. 2015, fig. 15 |
| Feather-damage porosity model | Interpolation anchored at 3 measured points (sealed gaps Cd≈0.30, intact Cd≈0.62, bare-cork limit) | Alam, Chowdhury et al., *Effect of Porosity of Badminton Shuttlecock on Aerodynamic Drag*, Procedia Engineering (2015); Kitta et al. (2011); Cooke (1999, 2002) |
| Nylon skirt deformation (speed-dependent C_D) | Cross-section halves by 50 m/s | Cohen et al., *Shuttlecock velocity decay after smash and slice shots in badminton*, Physica Scripta (2026), doi:10.1088/1402-4896/ae5361 |
| Lateral (Magnus-like) force from asymmetric damage | Derived from the drag the missing vanes no longer carry; kept deliberately small — badminton's lateral aerodynamics were long thought not to exist at all | Cohen et al., *Left- vs right-handed badminton slice shots*, C. R. Physique **25**, 1 (2024) |
| Valid Reynolds range for constant-C_D | `Re = 1.3×10⁴ – 2×10⁵`; flights that spend real time above it are flagged | Cooke (1999) |

Every one of these is documented in the source with a comment explaining *why*,
not just what — including two cases where an earlier version of the model got
the physics backwards (feather damage that increased drag instead of decreasing
it; a planar-model rotation term that turned asymmetric damage into steady lift)
and the literature-based argument for the fix, both left in place as comments so
the reasoning is auditable, not just the current numbers.

## Running it

**Full app (Gradio, needs a server):**

```bash
pip install gradio numpy pandas plotly scipy
python shuttlecock_simulator_9.2.py
```

**No-server version (Jupyter / Google Colab):**

```python
%pip install -q ipywidgets plotly
import shuttlecock_ipywidgets as app
a = app.launch()
```

**Just the physics, no UI:**

```python
import importlib.util
spec = importlib.util.spec_from_file_location(
    "shuttlecock", "shuttlecock_simulator_9.2.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

print(S.ValidationEngine().run_all())
print(S.physics_verification_suite(S.ParameterDatabase()))
```

## Getting the trajectories in the first place

The simulator predicts flights. `AIPhysicsTracker.py` is the other half: it turns
a video of a real shuttlecock flight into a per-frame trajectory that can be fed
into the physics analysis, together with the evidence needed to decide whether
that trajectory is good enough to use.

```bash
pip install gradio numpy pandas plotly scipy opencv-python
python AIPhysicsTracker.py     # set Profile = Shuttlecock in step 5
```

What makes it shuttlecock-specific rather than a generic object tracker:

- **One consistent centre definition.** A shuttlecock is a cork head plus a
  feather skirt, so "the position of the shuttlecock" needs defining. Four
  candidate definitions are computed (bbox centre, mask centroid, weighted
  centroid, cork centre) and one is chosen and then **held fixed for the whole
  run**, so damaged and undamaged shuttlecocks are measured the same way. The
  cork centre moves 0.00 px across the four damage conditions; the mask centroid
  moves up to 1.38 px.
- **Damage conditions as a prior.** `NORMAL`, `2_FEATHERS_SYMMETRIC`,
  `2_FEATHERS_ASYMMETRIC`, `4_FEATHERS_SYMMETRIC`, `4_FEATHERS_ASYMMETRIC` widen
  the appearance and shape tolerances only, leaving the motion-continuity checks
  untouched.
- **Detection separated from tracking.** YOLO and the OpenCV trackers only
  propose candidates; a candidate that does not fit the trajectory is rejected
  however confident the detector was. Recovery after a failure must also be
  physically reachable — a match farther away than the elapsed frames allow, or
  on the wrong side of the direction of flight, is refused rather than recorded.
- **Estimated values are never disguised as observations.** Every row carries
  `source` (`DETECTED` / `RECOVERED` / `PREDICTED` / `INTERPOLATED` / `MANUAL`)
  and `tracking_status` (`HIGH` / `MEDIUM` / `LOW` / `LOST`). Predicted rows can
  never reach HIGH or MEDIUM, and gaps longer than a few frames are left empty
  instead of being filled in.
- **A validation mode that answers the question with numbers.** Give it a
  reference trajectory and it reports position MAE/RMSE, success and lost-frame
  rates, velocity error, and the relative error of v0, flight time, range, max
  height, mean speed, speed decay, beta and C_D — then judges the run against
  fixed thresholds and says which criterion failed.

Measured performance on synthetic footage with an exact ground-truth trajectory
(7 required conditions + 3 hard ones), and the reasoning behind each design
choice:

- [`docs/shuttlecock_tracking.md`](docs/shuttlecock_tracking.md) — pipeline,
  confidence, recovery and validation methods
- [`docs/shuttlecock_tracking_results.md`](docs/shuttlecock_tracking_results.md)
  — measured errors, and the answer to "can this be used as research data?"

```bash
python3 tools/test_shuttlecock_tracking.py --verbose
```

## Scope and limitations

- The porosity (feather-damage) curve is an explicit interpolation between
  measured anchor points, not itself a measured Cd-vs-missing-feathers curve —
  the source says so, and the UI reports it as a model estimate rather than a
  measurement.
- Above Re ≈ 2×10⁵ (roughly a 300 km/h+ smash) the constant-C_D assumption is
  an extrapolation past Cooke's measured range; the simulator flags this rather
  than staying silent about it.
- 3D lateral/roll dynamics are model predictions, not validated against
  motion-tracked badminton footage — the UI labels them as such wherever they
  are shown.
- The tracker's reported accuracy comes from synthetic footage with an exact
  ground truth. Synthetic frames have a simple background and no turnover, lens
  distortion, compression noise or changing light. Accuracy on real experimental
  footage has to be established per camera setup by supplying a reference
  trajectory and running the validation mode — which is what that mode is for.

## License

MIT — see [`LICENSE`](LICENSE). You may use, modify and redistribute this,
including commercially, as long as the copyright notice travels with it. If you
use it in published work, a citation is appreciated but not legally required.

## Citing this repository

If you cite this simulator in a paper or report, please link to this
repository at the specific commit you used (the commit hash pins the exact
version, since the code may change after publication).
