# Design notes, C++ kernels, related work

## C++ kernels

`cpp/` is an optional C++17 package (nanobind bindings, no other dependencies) with the two hot
loops: the soft-finger stability solver and the analytic IK. Python uses it automatically when
installed (`pip install ./cpp`) and falls back to numpy otherwise; tests check both give the same
answers, and CI runs the suite both ways.

| Kernel | numpy | C++ | speed-up |
|---|---|---|---|
| stability, per evaluation (100k batch) | 1.08 µs | 0.44 µs | 2.4× |
| IK, per pose (16 q7 × 4 branches, 5k batch) | 91 µs | 14 µs | 6.7× (2.7× with MSVC on Windows) |
| IK, one grasp's 28 poses (per call) | 2.24 ms | 0.31 ms | 7.2× |
| whole check 2–5 pipeline, 6 objects | 10.1 s | 8.8 s | 1.15× |

Honest reading: numpy was already vectorised, so the kernels are 2–7× faster, but the pipeline
is dominated by MuJoCo collision queries and Python loops. The kernels matter where they would
in a robot: a dependency-free C++ core that can run inside a controller.

## Design notes (and the bugs that motivated them)

| Choice | Why |
|---|---|
| **Force-limited gripper** (`ctrl` = finger gap in m, tendon force clamped) | While the fingers are blocked by the object the servo saturates, so each finger pushes with a known force (20 N). The friction checks need the normal force; the Menagerie default (stiffness 100 N/m) squeezes with only ~1 N. |
| **Elliptic friction cones, `impratio = 10`** | MuJoCo contacts are soft, so "static" friction lets a held object creep. With the defaults the box crept ~1 mm/s in a 20 N grip. |
| **Torsional friction, `condim = 4`** | Lets a two-finger grasp resist torque about the finger axis (the soft-finger model). With `condim = 3` that torque cannot be resisted at all. |
| **Grasp twin (180° about the approach axis)** | A parallel gripper touches the same points either way, but the wrist (joint 7) differs by π. The first version drove joint 7 into its limit, the IK lost track, and the arm knocked the box into the palm. The planner now tries both twins and keeps the one with the larger joint-limit margin. |
| **Cartesian paths + one min-jerk profile per segment** | A joint-space interpolation between far-apart configurations swept the held box through the table. Paths are now straight TCP lines solved by IK at evenly spaced points. |
| **Arm gravity compensation** | As on the real Franka. The payload's weight is not compensated. |
| **TCP site 0.1034 m from the hand frame** | Midway between the fingertip pads, on the approach axis. Grasp poses are poses of this frame. |
| **QR patch as a separate visual geom (group 1)** | Lets the visibility check ray-cast against the label alone. |
| **Labelling hand welded to a mocap target** (not a mocap body itself) | A mocap body teleports with zero velocity, so MuJoCo's velocity-level friction cannot drag the held object along: the first version dropped every object. Welding a dynamic hand to the mocap target gives it a real velocity. |
| **Friction μ = 0.5** | Plastic fingertips on cardboard. With μ = 1 almost every candidate held, which made the labels uninformative. |
| **Stiffer finger/object contacts (5 ms instead of 20 ms)** | MuJoCo's regularised friction made held objects creep: 13 mm in a gentle 2.5 s transit, inversely proportional to its duration. 5 ms (still ≥ 2 × timestep) cut it 5× and lifted the stability model's AUC from 0.8 to 0.92–0.94. |
| **Two QR decoders** | OpenCV's classic detector missed a fully visible 90-px code on busy packaging; its ArUco-based detector read it. Real scanners are robust, so the judge tries both. |
| **Gross-slip threshold 20 mm / 15°** | A 70 g box with 96 % friction reserve still creeps a few mm; a 5 mm limit would score the simulator, not the grasp. Exact slip is always recorded. |
| **Target configuration closest to the start one** | The checker first picked the target configuration with the most joint-limit margin; the arm then spun its wrist 265° while carrying the object. |
| **One live offscreen renderer per process** | Freeing a MuJoCo renderer deletes GL objects in whichever context is current. When an old episode object was garbage-collected after a new one had made its renderer, the new scanner image became noise (Linux) or wrong (Windows) and counted as an unreadable QR. Found because one variant failed 68 times on "QR"; a guard now rejects noise images. |
| **Collision-checked joint-space segments with detours** | The transit swung held objects into the table; the planner now checks the path and detours via a raised pose. |

## Related work

- Target-pose-conditioned grasp feasibility has been learned before: Liu et al. predict IK and
  collision feasibility of a grasp for a given start and target pose
  ([arXiv:2512.18922](https://arxiv.org/abs/2512.18922)); Qin et al. learn an energy-based
  model of grasp feasibility across poses for pick-and-place
  ([arXiv:2506.15920](https://arxiv.org/abs/2506.15920)); ReorientBot learns grasp/placement
  feasibility for specific-pose placement ([arXiv:2202.11092](https://arxiv.org/abs/2202.11092)).
  None of them model **visibility of a surface region** or **grasp stability under the rotated
  gravity at the target**, and none output a **dense goal-conditioned heatmap**: those are the
  focus here.
- Grasp quality maps: GG-CNN, Contact-GraspNet, Graspness. Intent-dependent contact maps: ContactDB.
- Labelling recipe: ACRONYM (Eppner et al., 2021), here rebuilt in MuJoCo on open objects.


## Roadmap

- [x] Scene, QR object, scanner camera, scripted pick → move → show, ground-truth evaluation
- [x] Real objects (40 scanned products), automatic QR placement, grasp sampler, simulated lift labels
- [x] Analytic stability (checks 1–2): soft-finger model, validated on 5,225 simulated grasps
- [x] Reachability at start and target (checks 3–4) · QR visibility by ray casting (check 5)
- [x] C++ kernels (stability, analytic IK) with Python bindings and speed comparison
- [x] Validate checks 1–5 against full pick-and-show episodes (QR decoded by OpenCV)
- [x] Target *set* (12 rotations about the viewing axis)
- [x] Robustness to object-pose error (evaluation, 3 noise levels)
- [ ] Robust labels: train on outcomes under pose noise
- [x] Baseline (most stable grasp) vs target-aware selection, ablations: success-rate table
- [x] Learned scorer (point cloud + grasp + target → success), trained on full-episode outcomes; speed and unseen-object test
- [x] Goal-conditioned grasp heatmaps on the object surface (way A: checks painted on the mesh)
- [x] Way B: dense goal-conditioned heatmap network, tested in full episodes on unseen objects
- [x] Part 2: transport motion that keeps the grasp safe: dynamic grasp model (AUC 0.88), optimiser (38 % faster where the grasp limits)
- [x] Part 2: transport network: trained on the constraints, 99 % in full episodes (= optimiser), 3 % shorter, 1 ms vs 25 s
- [x] Part 2: grasp chosen together with the motion (19 % shorter carries, 100 % success)
- [x] Part 3: 112 objects of many shapes, random scanner poses; way B retrained: 61 % (87 % on boxes) vs 47 % for the checks
- [x] Part 3: grip model with every finger contact; hook grasps shown hanging (hook vs pinch with weight)
- [ ] Part 3: turning an object so its handle can be reached; fine-tuning in simulation
- [ ] v2: dexterous hand · v3: regrasp · v4: unknown shape · v5: explore, then show

