# LIBERO Goal-Swap Promoted Recipes

This file contains executable policies promoted from completed ASPIRE fix
loops. Exact task matches take priority over generic templates. Copy the
matching program as a whole: partial rewrites commonly remove the live
post-grasp compensation that made the policy work.

## `put_the_bowl_on_the_stove`

Use this program unchanged. It is observation-only and contains no fixed world
coordinates or seed branches. Evidence: development seeds 51--65 passed 12/15;
the frozen policy passed held-out seeds 1--50 at 28/50 (56%). Qwen3.5-4B later
copied this recipe as a single code revision and passed 10/15 development plus
10/15 terminal replays in episode `9a2117e7-836a-4eaa-abe0-87a58da77134`.
Source:
`outputs/libero_fix_loop/libero_goal_swap/put_the_bowl_on_the_stove/fix_code.py`,
2026-09-15.

```python
import numpy as np
from scipy.spatial.transform import Rotation


def localize(rgb, depth, intrinsics, extrinsics, prompts):
    depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for candidate in sorted(
            masks, key=lambda item: item.get("score", 0.0), reverse=True
        ):
            mask = candidate["mask"].astype(np.uint8)
            points = mask_to_world_points(
                mask, depth_img, intrinsics, extrinsics
            )
            if points is not None and len(points) >= 10:
                center = get_oriented_bounding_box_from_3d_points(points)["center"]
                return center, points, mask
    raise RuntimeError("Could not localize prompts: " + ", ".join(prompts))


def topdown_quaternion(yaw):
    matrix = Rotation.from_euler("z", yaw).as_matrix() @ np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def localize_near(
    rgb, depth, intrinsics, extrinsics, prompt, expected, scale
):
    depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
    choices = []
    for candidate in segment_sam3_text_prompt(rgb, prompt)[:10]:
        points = mask_to_world_points(
            candidate["mask"].astype(np.uint8),
            depth_img,
            intrinsics,
            extrinsics,
        )
        if points is None or len(points) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(points)["center"]
        candidate_scale = float(
            max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
        )
        # SAM can return a tiny bowl fragment very close to the wrist.  Reject
        # it before the proximity ranking: it produces a false live offset.
        if candidate_scale < 0.5 * scale or candidate_scale > 1.6 * scale:
            continue
        distance = float(np.linalg.norm(center - expected))
        if distance < 2.0 * scale:
            choices.append((distance, center, points))
    if not choices:
        raise RuntimeError("Held bowl not found near the gripper")
    _, center, points = min(choices, key=lambda item: item[0])
    return center, points


obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
intrinsics, extrinsics = cam["intrinsics"], cam["pose_mat"]
bowl_center, bowl_points, _ = localize(
    rgb, depth, intrinsics, extrinsics,
    ["bowl", "silver bowl", "metal bowl"],
)
stove_center, stove_points, _ = localize(
    rgb, depth, intrinsics, extrinsics,
    ["stove burner", "burner", "stove", "hotplate"],
)

radial = stove_center[:2] - bowl_center[:2]
radial /= max(float(np.linalg.norm(radial)), 1e-6)
bowl_scale = float(
    max(np.ptp(bowl_points[:, 0]), np.ptp(bowl_points[:, 1]))
)
horizontal_radius = 0.45 * bowl_scale
grasp_position = np.array([
    bowl_center[0] + radial[0] * horizontal_radius,
    bowl_center[1] + radial[1] * horizontal_radius,
    float(np.percentile(bowl_points[:, 2], 35.0)),
])
grasp_quaternion = topdown_quaternion(
    float(np.arctan2(radial[1], radial[0]) - np.pi / 2.0)
)

open_gripper()
goto_pose(grasp_position, grasp_quaternion, z_approach=0.12)
goto_pose(grasp_position, grasp_quaternion)
close_gripper()
lift_position = grasp_position.copy()
lift_position[2] += 0.18
goto_pose(lift_position, grasp_quaternion)

held_obs = get_observation()
held_cam = held_obs["agentview"]
held_center, held_points = localize_near(
    held_cam["images"]["rgb"],
    held_cam["images"]["depth"],
    held_cam["intrinsics"],
    held_cam["pose_mat"],
    "bowl",
    lift_position,
    bowl_scale,
)

stove_top = float(np.percentile(stove_points[:, 2], 98.0))
wrist_above_bottom = lift_position[2] - float(
    np.percentile(held_points[:, 2], 2.0)
)
live_xy_offset = lift_position[:2] - held_center[:2]
release_position = np.array([
    stove_center[0] + live_xy_offset[0],
    stove_center[1] + live_xy_offset[1],
    stove_top + wrist_above_bottom + 0.012,
])
transit_position = release_position.copy()
transit_position[2] = max(
    lift_position[2], release_position[2] + 0.12
)
goto_pose(transit_position, grasp_quaternion)
goto_pose(release_position, grasp_quaternion)
open_gripper()
retreat_position = release_position.copy()
retreat_position[2] += 0.12
goto_pose(retreat_position, grasp_quaternion)
for _ in range(4):
    get_observation()
```

Do not replace the observed percentile heights with constants. Do not replace
the target-facing rim grasp with the highest-scoring GraspNet pose. Do not omit
the post-lift re-observation: the bowl can swing, so the live object-to-wrist
offset is part of the placement policy.

## Bowl Placement Transfer: Plate or Cabinet Top

For `put_the_bowl_on_the_plate`, copy the complete promoted stove program above
and make only these substitutions:

- replace `stove_center` / `stove_points` / `stove_top` with
  `plate_center` / `plate_points` / `plate_top`;
- replace the target prompt list with
  `["plate", "ceramic plate", "dish"]`;
- keep the target-facing rim grasp, post-lift `localize_near`, percentile-98
  target surface, live XY offset, wrist-above-bottom term, `0.012` clearance,
  transit, retreat, and settling calls unchanged.

For `put_the_bowl_on_top_of_the_cabinet`, use the same complete program with
target identifiers changed to `cabinet_center`, `cabinet_points`, and
`cabinet_top`, and prompts
`["cabinet top", "top of cabinet", "cabinet"]`. The target height must still
come from percentile 98 of observed target points; never use a fixed cabinet
height.

### Mandatory cabinet-top correction

Do **not** re-segment the cabinet after grasping the bowl.  In development
episode `8c7aa2d2-57a7-4a71-9653-19222a54e80b`, the initial `"dark cabinet"`
mask correctly localized the black cabinet at approximately
`[0.406, -0.234, 0.171]`, but the robot then occluded it.  Re-running SAM after
the lift selected the wooden rack instead, and every valid replay missed the
cabinet.  Freeze `cabinet_center`, `cabinet_points`, and `cabinet_top` from the
initial observation.  Only the held bowl is re-localized after the lift.

For the exact task `put_the_bowl_on_top_of_the_cabinet`, copy the complete
program below.  Do not redesign it, add fixed-coordinate fallbacks, or replace
three-dimensional positions with two-dimensional slices.

```python
import numpy as np
from scipy.spatial.transform import Rotation


def localize(rgb, depth, intrinsics, extrinsics, prompts):
    depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        for candidate in sorted(
            masks, key=lambda item: item.get("score", 0.0), reverse=True
        ):
            points = mask_to_world_points(
                candidate["mask"].astype(np.uint8),
                depth_img,
                intrinsics,
                extrinsics,
            )
            if points is not None and len(points) >= 10:
                center = get_oriented_bounding_box_from_3d_points(points)["center"]
                return center, points
    raise RuntimeError("Could not localize prompts: " + ", ".join(prompts))


def localize_near(
    rgb, depth, intrinsics, extrinsics, prompt, expected, scale
):
    depth_img = depth[:, :, 0] if depth.ndim == 3 else depth
    choices = []
    for candidate in segment_sam3_text_prompt(rgb, prompt)[:10]:
        points = mask_to_world_points(
            candidate["mask"].astype(np.uint8),
            depth_img,
            intrinsics,
            extrinsics,
        )
        if points is None or len(points) < 10:
            continue
        center = get_oriented_bounding_box_from_3d_points(points)["center"]
        candidate_scale = float(
            max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
        )
        # Cabinet retries can hold the bowl high and partially behind the
        # wrist.  The first successful retry produced a 0.178 gripper-width
        # signal but SAM exposed less than half of the original silhouette.
        if candidate_scale < 0.2 * scale or candidate_scale > 1.6 * scale:
            continue
        distance = float(np.linalg.norm(center - expected))
        if distance < 2.0 * scale:
            choices.append((distance, center, points))
    if not choices:
        raise RuntimeError("Held bowl not found near the gripper")
    _, center, points = min(choices, key=lambda item: item[0])
    return center, points


def topdown_quaternion(yaw):
    matrix = Rotation.from_euler("z", yaw).as_matrix() @ np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    )
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


obs = get_observation()
cam = obs["agentview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
intrinsics, extrinsics = cam["intrinsics"], cam["pose_mat"]
bowl_center, bowl_points = localize(
    rgb, depth, intrinsics, extrinsics,
    ["bowl", "silver bowl", "metal bowl"],
)
cabinet_center, cabinet_points = localize(
    rgb, depth, intrinsics, extrinsics,
    ["dark cabinet", "cabinet top", "cabinet"],
)
# Freeze the unobstructed target observation before moving the robot.
cabinet_top = float(np.percentile(cabinet_points[:, 2], 98.0))

radial = cabinet_center[:2] - bowl_center[:2]
radial /= max(float(np.linalg.norm(radial)), 1e-6)
bowl_scale = float(
    max(np.ptp(bowl_points[:, 0]), np.ptp(bowl_points[:, 1]))
)
# A target-facing rim grasp is not reachable in every cabinet-task layout.
# Try the other rim sides when the observed gripper closes on air.  The
# normalized gripper coordinate is part of the public observation: <=0.03 is
# empty closure, while successful bowl grasps in recorded traces are ~0.13.
grasp_position = None
grasp_quaternion = None
for angle_offset in [0.0, np.pi, np.pi / 2.0, -np.pi / 2.0]:
    c, s = np.cos(angle_offset), np.sin(angle_offset)
    direction = np.array([
        c * radial[0] - s * radial[1],
        s * radial[0] + c * radial[1],
    ])
    candidate_position = np.array([
        bowl_center[0] + direction[0] * 0.45 * bowl_scale,
        bowl_center[1] + direction[1] * 0.45 * bowl_scale,
        float(np.percentile(bowl_points[:, 2], 35.0)),
    ])
    candidate_quaternion = topdown_quaternion(
        float(np.arctan2(direction[1], direction[0]) - np.pi / 2.0)
    )
    open_gripper()
    goto_pose(candidate_position, candidate_quaternion, z_approach=0.12)
    goto_pose(candidate_position, candidate_quaternion)
    close_gripper()
    grasp_check = get_observation()
    if float(grasp_check["robot_cartesian_pos"][-1]) > 0.03:
        grasp_position = candidate_position
        grasp_quaternion = candidate_quaternion
        break
    open_gripper()
    retreat = candidate_position.copy()
    retreat[2] += 0.12
    goto_pose(retreat, candidate_quaternion)
if grasp_position is None:
    raise RuntimeError("All observed bowl rim grasps closed on air")

lift_position = grasp_position.copy()
lift_position[2] += 0.18
goto_pose(lift_position, grasp_quaternion)

held_obs = get_observation()
held_cam = held_obs["agentview"]
held_center, held_points = localize_near(
    held_cam["images"]["rgb"],
    held_cam["images"]["depth"],
    held_cam["intrinsics"],
    held_cam["pose_mat"],
    "bowl",
    lift_position,
    bowl_scale,
)
wrist_above_bottom = lift_position[2] - float(
    np.percentile(held_points[:, 2], 2.0)
)
live_xy_offset = lift_position[:2] - held_center[:2]
release_position = np.array([
    cabinet_center[0] + live_xy_offset[0],
    cabinet_center[1] + live_xy_offset[1],
    cabinet_top + wrist_above_bottom + 0.012,
])
transit_position = release_position.copy()
transit_position[2] = max(
    lift_position[2], release_position[2] + 0.12
)
goto_pose(transit_position, grasp_quaternion)
goto_pose(release_position, grasp_quaternion)
open_gripper()
retreat_position = release_position.copy()
retreat_position[2] += 0.12
goto_pose(retreat_position, grasp_quaternion)
for _ in range(4):
    get_observation()
```

These are controlled transfers of the promoted bowl policy, not permission to
redesign it. Validate each transferred task on development seeds before calling
it promoted. Never call `goto_home_joint_position()` while holding the bowl.

Scale filtering in `localize_near` is mandatory for the transferred policies.
Evidence from frozen bowl-to-plate episode
`c91efc78-5760-48d7-9f76-76de4ad98c68` (2026-09-21): failed seed 51 ranked a
99-point fragment closest to the wrist and placed the bowl on the plate edge;
successful seed 54 selected a full 1.4k--2.0k-point bowl and placed it near the
plate center. Comparing horizontal scale with the original bowl rejects the
fragment without using a fixed point-count threshold.
