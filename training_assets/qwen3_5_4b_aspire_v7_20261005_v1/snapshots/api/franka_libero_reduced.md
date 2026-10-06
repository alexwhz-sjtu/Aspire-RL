# Franka LIBERO reduced API (Online-RL V1)

Molmo is disabled and is not part of this contract.

get_observation() -> dict[str, typing.Any]
  Doc:
    Get the observation of the environment.
    Returns:
        observation:
            A dictionary containing the observation of the environment.
            The dictionary contains the following keys:
            - ["agentview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
            - ["agentview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32.
            - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
            - ["agentview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
            - ["robot0_eye_in_hand"]["images"]["rgb"]: Current wrist camera image as a numpy array of shape (H, W, 3), dtype uint8.
            - ["robot0_eye_in_hand"]["images"]["depth"]: Current wrist camera depth image as a numpy array of shape (H, W), dtype float32.
            - ["robot0_eye_in_hand"]["intrinsics"]: Wrist camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
            - ["robot0_eye_in_hand"]["pose_mat"]: Wrist camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
            - ["robot_cartesian_pos"]: Current end-effector (panda_hand) pose in the robot/world frame as a numpy array of shape (8,), dtype float64. The first 3 elements are the robot's end-effector XYZ, the next 4 elements are the quaternion wxyz, and the last element is the gripper position normalized, 0 (closed) to 1 (open).
            - ["robot_joint_pos"]: Current joint positions as a numpy array of shape (7,), dtype float64. The last element is the gripper position normalized, 0 (closed) to 1 (open).

segment_sam3_text_prompt(rgb: numpy.ndarray, text_prompt: str) -> list[dict[str, typing.Any]]
  Doc:
    Run SAM3 segmentation on an RGB image conditioned on a text prompt.
    
    Args:
        rgb:
            RGB image array of shape (H, W, 3), dtype uint8.
        text_prompt:
            Text prompt for SAM3 segmentation.
    
    Returns:
        masks:
            A list of dictionaries. Each dict may contain:
    
              - "mask":  np.ndarray of shape (H, W), dtype bool,
                          where True means the pixel belongs to the instance.
              - "box": list [x1, y1, x2, y2] in pixel coordinates.
              - "score": float confidence score.
    Note:
        Returns an empty list if no results are found.
    
    Example:
        >>> rgb = obs["agentview"]["images"]["rgb"]
        >>> masks = segment_sam3(rgb, text_prompt="red mug")

segment_sam3_point_prompt(rgb: numpy.ndarray, point_coords: tuple[float, float]) -> list[dict[str, typing.Any]]
  Doc:
    Run SAM3 segmentation on an RGB image, optionally conditioned on an image coordinate point prompt.
    
    Args:
        rgb:
            RGB image array of shape (H, W, 3), dtype uint8.
        point_coords:
            (x, y) pixel coordinates of the point prompt.
    
    Returns:
        masks:
            A list of dictionaries. Each dict may contain:
    
              - "mask":  np.ndarray of shape (H, W), dtype bool,
                          where True means the pixel belongs to the instance.
              - "score": float confidence score.
    
    Example:
        >>> rgb = obs["agentview"]["images"]["rgb"]
        >>> masks = segment_sam3_point_prompt(rgb, (100, 100))

plan_grasp(depth: numpy.ndarray, intrinsics: numpy.ndarray, segmentation: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
  Doc:
    Plan grasp candidates using Contact-GraspNet for a single instance.
    
    This is a thin wrapper around the Contact-GraspNet planner. It does not
    apply any camera/world transforms or TCP offsets: the caller is
    responsible for transforming the resulting grasp poses into the desired
    frame and applying TCP offsets if necessary.
    
    Args:
        depth:
            Depth image in meters.
            Shape: (H, W) or (H, W, 1), dtype float32/float64.
        intrinsics:
            Camera intrinsic matrix.
            Shape: (3, 3), dtype float64.
        segmentation:
            Instance segmentation map where each integer > 0 corresponds to a
            unique object instance ID.
            Shape: (H, W) or (H, W, 1), dtype int32/int64.
    
    Returns:
        grasp_poses:
            np.ndarray of shape (K, 4, 4), dtype float64.
            Homogeneous transforms for each candidate grasp IN THE CAMERA FRAME.
        grasp_scores:
            np.ndarray of shape (K,), dtype float64.
            Confidence score for each candidate grasp.
    
    Example:
        >>> cam = obs["agentview"]
        >>> rgb = cam["images"]["rgb"]
        >>> depth = cam["images"]["depth"][:, :, 0]
        >>> sam3_results = sam3_seg_fn(rgb, text_prompt="red mug")
        >>> best = max(sam3_results, key=lambda d: d["score"])
        >>> mask = best["mask"]
        >>> K = cam["intrinsics"]
        >>> grasp_sample_tf, grasp_scores = plan_grasp(
        ...     depth=depth,
        ...     intrinsics=K,
        ...     segmentation=mask,
        ... )
        >>> best_idx = grasp_scores.argmax()
        >>> best_T = grasp_poses[best_idx]  # (4, 4)
        >>> camera_extrinsics = cam["pose_mat"]
        >>> grasp_sample_world_frame = camera_extrinsics @ best_T

plan_grasp_from_point_clouds(pc_full: numpy.ndarray, pc_segment: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
  Doc:
    Plan grasp candidates using Contact-GraspNet given a full point cloud and a segmented point cloud of the object wanting to be grasped. These point clouds can be composed from multiple viewpoints.
    
    Args:
        pc_full:
            Point cloud of the full environment, including the object to be grasped.
            Shape: (N, 3), dtype float32.
        pc_segment:
            Point cloud of the segmented object to be grasped.
            Shape: (N, 3), dtype float32.
    Returns:
        grasp_sample_tf: (4, 4) homogeneous transform for the grasp pose in THE POINT CLOUD FRAME.
        grasp_scores: (N,) array of grasp scores.

get_oriented_bounding_box_from_3d_points(points: numpy.ndarray) -> dict[str, typing.Any]
  Doc:
    Get the oriented bounding box from 3D points.
    
    Args:
        points: np.ndarray: The 3D points to get the oriented bounding box from.
            Shape: (N, 3), dtype float64.
    
    Returns:
        dict[str, Any]: The oriented bounding box. The dictionary contains the following keys:
            - "center": np.ndarray: The center of the oriented bounding box in point cloud frame.
            - "extent": np.ndarray: The extent of the oriented bounding box.
            - "R": np.ndarray: The rotation matrix of the oriented bounding box in point cloud frame.
    
    Example:
        >>> points = np.random.randn((100, 3))
        >>> obb = get_oriented_bounding_box_from_3d_points(points)

solve_ik(position: numpy.ndarray, quaternion_wxyz: numpy.ndarray) -> numpy.ndarray
  Doc:
    Solve inverse kinematics for the panda_hand link.
    
    Args:
        position:
            Target position in world frame.
            Shape: (3,), dtype float64.
        quaternion_wxyz:
            Target orientation as a unit quaternion in world frame.
            Shape: (4,), [w, x, y, z], dtype float64.
    
    Returns:
        joints:
            np.ndarray of shape (7,), dtype float64.
            Joint angles for the 7 DoF Franka arm.
    
    Example:
        >>> target_pos = np.array([0.5, 0.0, 0.3])
        >>> target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity, wxyz
        >>> joints = solve_ik(target_pos, target_quat)
        >>> move_to_joints(joints)

move_to_joints(joints: numpy.ndarray) -> None
  Doc:
    Move the robot to a given joint configuration in a blocking manner.
    Interpolation is slightly rudimentary so it is recommended to keep pre-grasp cartesian offsets smaller e.g. 0.075m.
    
    Args:
        joints:
            Target joint angles for the 7-DoF Franka arm.
            Shape: (7,), dtype float64.
    
    Returns:
        None
    
    Example:
        >>> joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]) # this is an example home joint configuration for the Franka arm
        >>> move_to_joints(joints)

open_gripper() -> None
  Doc:
    Open gripper fully.
    
    Args:
        None

close_gripper() -> None
  Doc:
    Close gripper fully.
    
    Args:
        None

goto_pose(position: numpy.ndarray, quaternion_wxyz: numpy.ndarray, z_approach: float = 0.0) -> None
  Doc:
    Go to pose using Inverse Kinematics with optional approach motion.
    
    Combines solve_ik + move_to_joints into a single call. If z_approach > 0,
    first moves to position offset by z_approach in the gripper's Z axis,
    then moves to the final position.
    
    Args:
        position: (3,) XYZ target in meters (world frame).
        quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
        z_approach: Z offset for approach motion (meters). Default 0.0.
    
    Example:
        >>> goto_pose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 1.0, 0.0, 0.0]), z_approach=0.075)

goto_home_joint_position() -> None
  Doc:
    Return the arm to its reset joint configuration with high manipulability

subsample_point_cloud(pc: numpy.ndarray, max_points: int = 10000) -> numpy.ndarray
  Doc:
    Randomly subsample a point cloud to a maximum number of points.
    
    Args:
        pc: (N, 3) array of points.
        max_points: The maximum number of points to subsample to. Default is 10000.
    Returns:
        subsampled_pc: (M, 3) array of subsampled points where M <= max_points.

filter_noise(points: numpy.ndarray, colors: numpy.ndarray | None = None) -> tuple[numpy.ndarray, numpy.ndarray | None]
  Doc:
    Filter noise from the point cloud.
    Args:
        points: (N, 3) array of points.
        colors: (N, 3) array of colors. Optional, default is None.
    Returns:
        (M, 3) array of points.
        (M, 3) array of colors. Optional, default is None.

rotation_matrix_to_quaternion(R: numpy.ndarray) -> numpy.ndarray
  Doc:
    Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].
    
    Implements the robust Sheppard's method (checking trace and diagonal elements)
    to avoid numerical instability when the trace is close to zero.
    
    Args:
        R: (3, 3) rotation matrix.
    
    Returns:
        np.array: [w, x, y, z] unit quaternion.

decompose_transform(T: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]
  Doc:
    Decompose a 4x4 homogeneous transformation matrix into position and quaternion.
    
    Args:
        T: (4, 4) homogeneous transformation matrix.
    
    Returns:
        tuple:
            - position: (3,) np.array
            - quaternion: (4,) np.array [w, x, y, z]

depth_to_point_cloud(depth_img: numpy.ndarray, intrinsics: numpy.ndarray) -> numpy.ndarray
  Doc:
    Convert a depth image to a 3D point cloud in the Camera Frame.
    
    Args:
        depth_img: (H, W) depth map in meters.
        intrinsics: (3, 3) camera intrinsic matrix.
    
    Returns:
        np.array: (H, W, 3) image of 3D coordinates.

mask_to_world_points(mask: numpy.ndarray, depth: numpy.ndarray, intrinsics: numpy.ndarray, extrinsics: numpy.ndarray) -> numpy.ndarray
  Doc:
    Convert specific pixels defined by a binary mask into 3D points in the World Frame.
    
    Args:
        mask: (H, W) binary mask (0 or 1).
        depth: (H, W) depth map.
        intrinsics: (3, 3) camera intrinsics.
        extrinsics: (4, 4) camera-to-world pose matrix.
    
    Returns:
        np.array: (N, 3) array of valid 3D points in world coordinates.

pixel_to_world_point(u: int, v: int, z: float, intrinsics: numpy.ndarray, extrinsics: numpy.ndarray) -> numpy.ndarray
  Doc:
    Deproject a single pixel to a 3D world point.
    
    Args:
        u, v: Pixel coordinates (col, row).
        z: Depth at that pixel.
        intrinsics: (3, 3) matrix.
        extrinsics: (4, 4) matrix.
    
    Returns:
        np.array: [x, y, z] in world frame.

transform_points(points: numpy.ndarray, transform_matrix: numpy.ndarray) -> numpy.ndarray
  Doc:
    Apply a 4x4 homogeneous transform to a set of 3D points.
    
    Args:
        points: (N, 3) or (H, W, 3) array of points.
        transform_matrix: (4, 4) homogeneous transformation matrix.
    
    Returns:
        np.array: Transformed points with same shape as input.

interpolate_segment(p1: numpy.ndarray, p2: numpy.ndarray, step: float = 0.03) -> list[numpy.ndarray]
  Doc:
    Generate waypoints along a line segment between two 3D points.
    
    Args:
        p1: Start point (3,).
        p2: End point (3,).
        step: Distance between waypoints in meters.
    
    Returns:
        list[np.ndarray]: List of points including p1 and p2.

normalize_vector(v: numpy.ndarray) -> numpy.ndarray
  Doc:
    Normalize a vector to unit length.
    
    Args:
        v: (3,) vector.
    
    Returns:
        np.array: (3,) unit vector.

select_top_down_grasp(grasps: numpy.ndarray, scores: numpy.ndarray, cam_to_world: numpy.ndarray, vertical_threshold: float = 0.8) -> tuple
  Doc:
    Selects the best grasp that aligns the gripper vertically (Top-Down).
    
    Args:
        grasps: (N, 4, 4) Grasp poses in camera frame.
        scores: (N,) Grasp scores.
        cam_to_world: (4, 4) Extrinsics matrix.
        vertical_threshold: Dot product threshold (1.0 is perfectly vertical).
    
    Returns:
        tuple: (best_grasp_world_matrix, best_score) or (None, -inf)
