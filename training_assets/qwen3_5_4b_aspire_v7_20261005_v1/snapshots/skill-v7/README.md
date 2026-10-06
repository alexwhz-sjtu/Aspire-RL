# LIBERO Shared Robot Skills

These files contain reusable, observation-only robot-control knowledge shared across LIBERO experiments. Pipeline orchestration belongs in experiment directories; measurement procedures belong in [../analysis/](../analysis/).

## Mandatory Executable-Program Contract

Read this before copying any motion template. A replay can return `sandbox_rc=0`
while doing nothing when the file only defines functions and never calls them.

1. Prefer a **flat top-level program**. Its first environment operation should be
   `obs = get_observation()`; do not hide the whole program inside `run()`.
2. If a wrapper function is unavoidable, the final non-comment line must call it,
   for example `run()`.
3. Never redefine or shadow an approved API name such as `get_observation`,
   `solve_ik`, `goto_pose`, or `plan_grasp`.
4. Before replay, check that every variable is assigned on every branch and that
   every call matches the frozen API reference. Do not submit placeholders.
5. On the first replay, inspect the trace summary. `api_0_calls == 0` means the
   program did not execute robot APIs; fix the entry point or early control flow
   before changing perception or motion geometry.
6. Validate one seed first. Only batch additional seeds after one replay has
   nonzero API calls; otherwise a batch wastes the debugging budget.
7. Never replace failed perception with guessed pixels, guessed depth, or fixed
   world coordinates. Localize named objects with SAM3 and observed depth.
8. For pick-and-place, preserve this causal order:
   `observe/localize pick+target -> open -> approach -> close -> lift -> move to
   target -> descend -> open`. There must be no `open_gripper()` between grasp
   closure and target release.
9. The localized target must numerically affect `above_target` and
   `release_pos`. Computing a target position and then moving relative to the
   pick position is not placement.
10. Match API field names literally. SAM3 results use `mask`, `box`, and
    `score`; there is no `bbox` field. Do not add a filter on a field that the
    frozen API reference does not define.

## High-Priority Task Routing

Before writing generic code, check whether the task has a promoted recipe.
For any Goal-Swap task that moves the bowl to the stove, plate, or cabinet top,
read [goal-swap.md](goal-swap.md) completely before writing code. The promoted
bowl policy passed 12/15 development and 28/50 held-out seeds in its source fix
loop, and Qwen3.5-4B copied it into a one-revision program that independently
passed 10/15 development and 10/15 terminal replays on 2026-09-21. A small
model must not replace that evidence-backed policy with a generic GraspNet
template or guessed heights.

For the exact cabinet-top task, copy the complete cabinet program in
`goal-swap.md`.  Freeze the cabinet pose from the unobstructed initial
observation; after the robot lifts the bowl it occludes the cabinet and a new
SAM query can select the wooden rack.  Re-localize only the held bowl.

Evidence: the frozen program for
`libero_goal_swap/put_the_bowl_on_the_stove` in online-RL episode
`ca6a4a0f-e380-44a4-b106-6ca36a7688d0` defined but never called `run()` and
shadowed `get_observation`. All development seeds 51--65 finished with
`sandbox_rc=0`, zero API calls, and zero task completions (2026-09-21).

Follow-up evidence: episode `62f6f715-556f-4c08-804b-e947c80d4886`
fixed liveness but froze code that guessed center pixels/depth, never used its
computed stove position, and opened the gripper before transport. It produced
real API calls but still scored 0/15 on development seeds (2026-09-21).

| Skill | Purpose |
|---|---|
| [goal-swap.md](goal-swap.md) | Promoted, executable Goal-Swap recipes; check this first for an exact task match |
| [execution.md](execution.md) | Mandatory liveness checks and a flat executable pick/place template for small coding models |
| [grasp.md](grasp.md) | Grasp selection, orientations, gripper evidence, and pick/place skeletons |
| [localize.md](localize.md) | SAM3 prompting, disambiguation, and 3D localization helpers |
| [transport.md](transport.md) | Waypoints, collision-free transit, and placement approaches |
| [manipulation.md](manipulation.md) | Drawers, knobs, switches, pushing, and other contact tasks |

Promoted findings should include provenance: source suite/task, development seeds, held-out result when available, source code path, and date. Avoid universal claims from one scene or seed.
