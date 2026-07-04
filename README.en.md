<img src="docs/media/logo_unige.png" alt="University of Geneva — Faculty of Science, Department of Computer Science" width="360">

[Français](README.md) · **English**

# Vision-assisted object grasping — SO-101

![License: MIT](https://img.shields.io/badge/license-MIT-green) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![Built with LeRobot](https://img.shields.io/badge/built%20with-LeRobot-ff9800)

This repository gathers the code I developed for my **final-year bachelor's thesis
(TFE)**: a **complete modular perception → planning → control pipeline**, "from pixel
to grasp", that makes an SO-101 robotic arm **grasp objects**, built on top of
[LeRobot](https://github.com/huggingface/lerobot).

It does not replace the official resources. To **build and wire the robot** (parts, 3D
printing, assembly, basic teleoperation), see
[SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) and the
[LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101). This
repository assumes you **already have a working, teleoperable SO-101**, and implements
on top of it the whole "**see → decide where to grasp → execute**" chain.

| | |
|---|---|
| **Student** | Maxence Chanier |
| **Supervisor** | Guido Bologna |
| **Course** | Final-year bachelor's thesis in Computer Science (University of Geneva) |
| **Academic year** | 2025-2026 |
| **Robot** | SO-101 — 6 Feetech STS3215 servos (5 joints + gripper), **underactuated, 5 DOF** |
| **Cameras** | 3 × USB 1920×1080 — `cam_0`/`cam_1` in *eye-to-hand* stereo + `cam_2` *eye-in-hand* |
| **Machine** | MacBook Pro M4 (Apple Silicon, macOS) |

## Demonstration

The cube grasped among the other objects by the **modular pipeline** (stereo perception
→ grasp-angle planning → descent and torque-servoed close):

![Grasp by the modular pipeline, cube among several objects](docs/media/demo_pipeline.gif)

## What this project does

From three cameras and the robot state, the chain locates an object in 3D, chooses a
grasp point and angle suited to its geometry, then executes the grasp and the drop —
with a closed-loop refinement right before closing the gripper.

- 3D localization by **multi-camera stereo vision** (HSV color or Hugging Face
  *open-vocabulary* detector).
- **Automatic choice of the grasp point and angle** according to the object's geometry
  and its reachability by the arm.
- **Closed-loop refinement** with the wrist camera before closing the gripper, then a
  **torque-servoed grasp** (retry if the gripper closes on nothing).
- **Two methods compared**: modular pipeline and *imitation learning* (ACT) — see
  [Two approaches](#two-approaches).

The step-by-step details of the chain are described in
[How grasping works](#how-grasping-works).

## Status and known limitations

The objects tested (cube, cylinder, box, triangular prism, ball, Rubik's Cube):

![The tested objects](docs/memoire/images/objets.jpg)

Grasping is **reliable on an isolated object** of simple geometry. It was tested on
several shapes — cube, cylinder, rectangular box, triangular prism, ball and Rubik's
Cube — grasped and dropped repeatedly, often on the first try (the quantitative
measurements cover the cube and the cylinder; the multicolored Rubik's Cube is only
detected with the *open-vocabulary* HF detector). Still open:

- **cluttered scenes and occlusion** (active viewpoint selection, obstacle avoidance) —
  this is the direction still in progress;
- **perception robustness under variable lighting** (HSV calibration to redo depending
  on the conditions).

## Installation

Prerequisites: an assembled and teleoperable SO-101 (see links at the top), Python 3.12,
and the three cameras plugged in.

```bash
git clone https://github.com/machanier/tfe-lerobot-so101.git
cd tfe-lerobot-so101
./setup_env.sh          # creates the venv, clones + installs LeRobot, installs dependencies
source venv/bin/activate
```

## Usage

```bash
source venv/bin/activate

# Check the whole calibration (motors, cameras, hand-eye) + kinematic self-tests
python scripts/check_calibration.py

# Teleoperate / preview a camera
python scripts/teleoperate.py
python scripts/preview_camera.py --camera 0

# Perception only: calibrate the colors (once, under the final lighting) then run
python scripts/calibrate_hsv.py
python scripts/run_perception.py                                  # live, 3 cameras
python scripts/run_perception.py --mode replay --replay <dataset>  # no robot, on recorded data

# Full pick-and-place (perception → grasp → drop)
python scripts/pick_and_place.py --target orange_cube --detector hsv
python scripts/pick_and_place.py --target orange_cube --detector hf   # open-vocabulary detector
```

By default, `pick_and_place.py` saves **diagnostic snapshots** (camera views at grasp
time) in `outputs/perception/`. The `--display` option also opens a live tracking
window; `--no-snapshots` disables saving.

## How grasping works

The three cameras at grasp time — `cam_0`/`cam_1` in stereo (detection and 3D position
of the object) and the `cam_2` wrist camera (close-up view):

![View of the three cameras during a grasp](docs/media/display_3cams.jpg)

The chain is orchestrated by [`src/pipeline.py`](src/pipeline.py):

1. **Perception** — HSV (or HF) detection in `cam_0`/`cam_1`, stereo triangulation of
   the 3D position (base frame, `z = 0` on the plate). A measured calibration bias is
   subtracted ([`configs/perception/bias_correction.json`](configs/perception/bias_correction.json)).
2. **Planning** ([`src/planning/grasp.py`](src/planning/grasp.py)) — candidate top-down
   / diagonal / front-facing angles proposed according to the zone, keeping the first one
   reachable by IK; approach aligned with the long axis for an elongated object.
3. **`cam_2` refinement** ([`src/control/closed_loop.py`](src/control/closed_loop.py)) —
   a few cm above the object, the *eye-in-hand* camera corrects the position and realigns
   the jaws, under safeguards (blob size, correction ceiling).
4. **Grasp offsets** — the commanded tool frame is not the point where the jaws clamp
   (the mechanism is mounted beside the wrist axis); two horizontal offsets (adaptive
   lateral = ½ width + margin; depth) bring the fingers to the right place.
5. **Grasp** — descent, torque-servoed close, check after lifting, retry if it closed on
   nothing, then drop.

The `cam_2` refinement in action — the wrist camera reframes the object and realigns the
jaws just before the descent:

<p align="center">
  <img src="docs/media/refine1.jpg" width="360" alt="cam_2 refinement (view 1)">
  <img src="docs/media/refine2.jpg" width="360" alt="cam_2 refinement (view 2)">
</p>

## Two approaches

The project compares two ways of solving the same task:

- **Modular pipeline** *(core of `src/`)* — explicit perception, rule-based geometric
  planning, inverse kinematics, closed loop. Interpretable and training-data-free.
- **Imitation learning (ACT)** — grasping is learned from teleoperated demonstrations,
  via the official LeRobot stack (`LeRobotDataset` + ACT policy). Detailed procedure in
  [`docs/IL_ACT_RUNBOOK.md`](docs/IL_ACT_RUNBOOK.md) and, for cloud-GPU training,
  [`docs/IL_COLAB.md`](docs/IL_COLAB.md).

![Grasp by the ACT policy (imitation learning)](docs/media/demo_politique.gif)

Datasets and trained models, public on the Hugging Face Hub:
- Dataset: [`Machanier/so101_orange_cube`](https://huggingface.co/datasets/Machanier/so101_orange_cube)
  (low-resolution variant: [`_lowres`](https://huggingface.co/datasets/Machanier/so101_orange_cube_lowres))
- ACT model: [`Machanier/act_so101_orange_cube`](https://huggingface.co/Machanier/act_so101_orange_cube)
  (low-resolution variant: [`_lowres`](https://huggingface.co/Machanier/act_so101_orange_cube_lowres))

## Results

Pick-and-place evaluation campaign (success rate, 95% Wilson confidence interval). Raw
data in [`results/`](results/), detailed analysis in the thesis.

| Object | Pipeline — HF | Pipeline — HSV | Imitation (ACT) |
|---|---|---|---|
| Cube *(30 trials/detector)* | **77%** | 67% | 70% |
| Cylinder *(9 trials/detector)* | **78%** | 67% | 0/5 *(generalization probe)* |

The two approaches are **tied on the cube** (overlapping confidence intervals); the tie
is broken on **generalization**, where the pipeline keeps the edge (cylinder,
language-designated objects) while the learned policy stays on its training distribution.

## Repository structure

```
tfe-lerobot-so101/
├── src/                  # the perception → planning → control pipeline
│   ├── perception/       # 2D detection + 3D reconstruction (stereo, PnP)
│   ├── planning/         # grasp point and angle selection (adaptive)
│   ├── control/          # 5-DOF IK, trajectories, cam_2 closed loop, motors
│   ├── calibration/      # hand-eye, forward kinematics, motors → angles
│   ├── utils/            # SE(3) helpers
│   └── pipeline.py       # end-to-end orchestration
├── scripts/              # CLI: calibration, perception, pick-and-place, IL (train/eval)
├── configs/              # calibrations of my setup + robot URDF model
├── tests/                # synthetic integration tests + grasp selection
├── hardware/             # 3D-printed models (structure, box, gripper, objects)
├── results/              # campaign CSVs (pipeline & IL grasping, structure, lighting)
├── docs/                 # base frame, IL runbook, thesis (PDF + LaTeX source), media
├── requirements.txt · setup_env.sh
└── LICENSE
```

Base frame and measurement procedure: [`docs/REPERE_BASE.md`](docs/REPERE_BASE.md).

## Thesis

This repository accompanies my bachelor's thesis, which details the approach, the design
choices and the evaluation:
**[Thesis — PDF](docs/memoire/Memoire_Bachelor_Chanier_Maxence.pdf)** (in French).

The **specifications** (official subject and objectives of the thesis):
[docs/cahier_des_charges.pdf](docs/cahier_des_charges.pdf) (in French).

## Resources

- [LeRobot](https://github.com/huggingface/lerobot) — Hugging Face stack (teleoperation,
  datasets, policies) this project builds on.
- [LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101)
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) — official hardware repository
  for the arm (source of the URDF).

## How to cite

```bibtex
@mastersthesis{chanier2026saisie,
  author = {Chanier, Maxence},
  title  = {Saisie d'objets en environnement pour un bras robotique assisté par vision},
  school = {Université de Genève, Faculté des sciences},
  year   = {2026},
  type   = {Travail de fin d'études}
}
```

## Development assistance

The code and documentation were developed with the help of an AI assistant (**Claude**,
via Claude Code). The design, the technical choices, the tuning on the robot and all the
decisions are the author's.

## License

Code distributed under the **MIT** license — see [LICENSE](LICENSE).
