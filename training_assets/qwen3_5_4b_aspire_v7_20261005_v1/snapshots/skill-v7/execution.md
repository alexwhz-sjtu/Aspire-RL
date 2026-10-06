---
name: execution
description: Mandatory program-liveness contract and a flat, directly executable pick-and-place template optimized for small coding models.
---

# Execution — Make the Robot Program Actually Run

## Non-Negotiable Liveness Rules

- Submit a complete Python **program**, not a library of uncalled functions.
- Prefer flat top-level statements. Start with `obs = get_observation()`.
- Never define a function with the same name as an approved API.
- Use only names and signatures present in the frozen API reference.
- Replay seed 51 alone first. If its trace reports `api_0_calls == 0`, repair
  program liveness before doing any geometry tuning.
- Do not batch many seeds until the program has produced nonzero API calls once.
- **Do not redesign the template before the first replay.** For a pick/place
  task, copy the flat template and initially change only `PICK_TEXT` and
  `TARGET_TEXT` to match the task language.
- Never guess image pixels, depth values, or world coordinates. A guessed call
  such as `pixel_to_world_point(width//2, height//2, 0.25, ...)` is invalid
  strategy even if Python executes successfully.

## Flat Pick-and-Place Template

Replace the two prompt strings with concrete objects from the task instruction.
Keep the program flat; do not wrap it in `run()`.

```python
import numpy as np

PICK_TEXT = "bowl"
TARGET_TEXT = "stove"

# Observe and localize both objects before the arm can occlude them.
obs = get_observation()
cam = obs["agentview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
depth_img = depth[:, :, 0] if len(depth.shape) == 3 else depth
K = cam["intrinsics"]
E = cam["pose_mat"]

pick_masks = segment_sam3_text_prompt(rgb, PICK_TEXT)
if not pick_masks:
    raise RuntimeError("pick object not found")
pick_result = max(pick_masks, key=lambda item: item["score"])
pick_mask = pick_result["mask"].astype(np.int32)

target_masks = segment_sam3_text_prompt(rgb, TARGET_TEXT)
if not target_masks:
    raise RuntimeError("placement target not found")
target_result = max(target_masks, key=lambda item: item["score"])
target_points = mask_to_world_points(
    target_result["mask"].astype(np.uint8), depth_img, K, E
)
if target_points is None or len(target_points) < 10:
    raise RuntimeError("placement target has too few 3D points")
target_center = get_oriented_bounding_box_from_3d_points(target_points)["center"]
target_top = float(target_points[:, 2].max())

# Plan a grasp. plan_grasp returns camera-frame transforms.
grasp_poses, grasp_scores = plan_grasp(depth_img, K, pick_mask)
if grasp_poses is None or len(grasp_poses) == 0:
    raise RuntimeError("no grasp candidates")
grasp_world, _ = select_top_down_grasp(grasp_poses, grasp_scores, E)
if grasp_world is None:
    grasp_world = E @ grasp_poses[int(np.argmax(grasp_scores))]
grasp_pos, grasp_quat = decompose_transform(grasp_world)

# Execute pick, lift, transport, and release.
open_gripper()
goto_pose(grasp_pos, grasp_quat, z_approach=0.075)
close_gripper()
lift_pos = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.15])
goto_pose(lift_pos, grasp_quat)
above_target = np.array([target_center[0], target_center[1], lift_pos[2]])
goto_pose(above_target, grasp_quat)
release_pos = np.array([target_center[0], target_center[1], target_top + 0.04])
goto_pose(release_pos, grasp_quat)
open_gripper()
for _ in range(3):
    get_observation()
```

Do not reorder those gripper operations. From `close_gripper()` through arrival
at `release_pos`, the gripper must remain closed. `target_center` and
`target_top` must be used to build the actual transport/release positions.

## Pre-Replay Checklist

- The file contains at least one reachable approved API call.
- No approved API name appears after `def`.
- There is no uncalled whole-program wrapper.
- Pick and target prompts are concrete strings, not `<placeholders>`.
- Variables such as `grasp_world`, `grasp_pos`, and `grasp_quat` are assigned on
  all branches before use.
- `segment_sam3_text_prompt` receives `(rgb, text)`—never depth or camera matrices.
- `plan_grasp` receives `(depth, intrinsics, segmentation)`.
- No fixed pixel/depth/world-coordinate fallback appears in the program.
- The target localization is used in both lateral transport and release.
- There is exactly one grasp close and no open until the target release.
- A `print("success")` statement is not validation and should not replace
  observation/replay evidence.

## Failure Classification

| Evidence | Meaning | Next change |
|---|---|---|
| `sandbox_rc != 0`, zero API calls | Python/sandbox failure before execution | Read stderr; fix syntax, unsafe constructs, or undefined names |
| `sandbox_rc == 0`, zero API calls | Uncalled entry point or empty control path | Flatten the program or call the wrapper |
| Nonzero API calls, no completion | Real perception/manipulation failure | Inspect trace/keyframes and tune prompts or motion |

## Pick-and-Place Dataflow Invariants

Before `final_submission`, trace the variables—not the comments:

```text
pick SAM3 mask -> observed pick depth -> grasp pose -> close_gripper
target SAM3 mask -> observed target depth -> target_center/target_top
target_center/target_top -> above_target/release_pos -> open_gripper
```

If any arrow is absent, the program is not implementing the language goal.
Never infer correctness from comments such as "move to stove" or from printing
"success"; only executed coordinates and replay reward count.
