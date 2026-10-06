---
name: transport
description: Motion patterns for moving objects between locations — multi-step waypoints, safe transit sequences, interpolated Cartesian moves, collision avoidance during transport. Grows through experiment.
---

# Transport — Motion Patterns

> This skill covers **how to move** once an object is grasped: waypointing, safe transit,
> intermediate stops, and collision avoidance. Discovered through experiment — add entries as
> you validate them.

---

## Waypoint Sequences

Add entries here when you find a multi-step motion pattern that prevents collisions, slip, or
IK failures during transport. Include: trigger condition, waypoint sequence, why it works.

| Pattern | Trigger | Sequence | Notes |
|---|---|---|---|

---

## Pre-Probe IK Conditioning

Before grasping, pre-visit the placement target at multiple approach heights. This seeds the
IK solver so it stays on the correct branch when placing after the grasp.

```python
# BEFORE grasping, probe the target at decreasing heights:
for probe_z in [target_z + 0.08, target_z + 0.06, target_z + 0.04, target_z + 0.02]:
    goto_pose(np.array([target_x, target_y, probe_z]), TOP_DOWN_QUAT)
goto_home_joint_position()
# NOW grasp — IK is conditioned for placement
```

**Why it works**: IK solvers maintain local branch continuity from their last configuration.
Visiting the placement target before grasping seeds the solver on the branch that can reach
that XY at low Z — so after grasp + home reset, the arm finds the same branch.

**When to use**: Any task where the placement target is at low Z and has a fixed known XY.

---

## Interpolated Cartesian Motion

Add entries here when fine-grained step-by-step moves outperform direct `goto_pose` (e.g. for
fragile placements, long horizontal transport, or constrained workspaces).

---

## Safe Transit (Lateral Escape Before Lift)

Add entries here when the arm must move laterally before lifting to avoid sweeping through
obstacles (opened drawers, adjacent objects, cabinet edges).

---

## Placement Approach

Add entries here for approach sequences above the target (hover height, descent speed,
drop vs. lower) that prevent bounce, tip, or miss.

### Live Post-Lift Offset and Bottom-Height Compensation

**Trigger:** A container is pinched at one rim and can swing or rotate after lift. A rigid
object-to-gripper offset measured before the grasp then places the container beside or above the
target. Re-localize the held object near the current gripper and use its live offset and bottom
height to construct the release pose.

```python
held_obs = get_observation()
held_cam = held_obs["agentview"]
held_center, held_points = localize_held_near_gripper(
    held_cam["images"]["rgb"], held_cam["images"]["depth"],
    held_cam["intrinsics"], held_cam["pose_mat"],
    "<held object prompt>", lift_position, object_scale,
)
if held_center is None:
    raise RuntimeError("held object could not be localized near gripper")
live_xy_offset = lift_position[:2] - held_center[:2]
wrist_above_bottom = lift_position[2] - np.percentile(held_points[:, 2], 2.0)
release_position = np.array([
    target_center[0] + live_xy_offset[0],
    target_center[1] + live_xy_offset[1],
    target_top + wrist_above_bottom + clearance,
])
```

**Why it works + evidence:** The live observation measures motion caused by contact dynamics
instead of assuming the grasped object is rigidly attached to the wrist. This compensation
contributed to successes on development seeds 52, 53, 54, 56, 57, 59, 60, 61, 62, 64, and 65
for `libero_goal_swap/put_the_bowl_on_the_stove`. Source:
`outputs/libero_fix_loop/libero_goal_swap/put_the_bowl_on_the_stove/fix_code.py`, 2026-09-15.

### Physics Settling After Release

After releasing a grasped object, call `get_observation()` several times to give the physics
engine time to settle the object before the episode's reward predicate fires.

```python
open_gripper()
for _ in range(3):
    get_observation()
```

**NEVER call `goto_home_joint_position()` while holding a grasped object.** Going home
during transport opens the arm configuration and drops the object. Transport in one continuous
arc: lift → translate → descend → release.

| Object type | Transit height | Release height | Notes |
|---|---|---|---|
