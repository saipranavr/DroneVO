import cv2
import sys
import numpy as np # Needed for converting points later

def detect_features(frame, detector):
    """Detects keypoints and computes descriptors for a given frame."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_frame, None)
    return gray_frame, keypoints, descriptors

def match_features(kp1, desc1, kp2, desc2, matcher):
    """Matches features between two sets of descriptors using Lowe's ratio test."""
    good_matches = []
    pts1 = []
    pts2 = []
    if desc1 is not None and desc2 is not None:
        matches = matcher.knnMatch(desc1, desc2, k=2)
        # Apply Lowe's ratio test
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx].pt)
                pts2.append(kp2[m.trainIdx].pt)
    # Convert points to float32 as required by findEssentialMat
    return good_matches, np.float32(pts1), np.float32(pts2)

def estimate_pose(pts1, pts2, camera_matrix, prob=0.999, threshold=1.0):
    """Estimates the relative camera pose from matched points."""
    if pts1 is None or pts2 is None or len(pts1) < 5 or len(pts2) < 5:
        # print("Warning: Not enough points for pose estimation.") # Optional print
        return None, None, None

    # Calculate Essential Matrix using RANSAC
    # Ensure pts1 and pts2 are in the correct shape (N, 1, 2) or (N, 2)
    # findEssentialMat expects points in shape (N, 2)
    E, mask = cv2.findEssentialMat(pts2, pts1, camera_matrix, method=cv2.RANSAC, prob=prob, threshold=threshold)

    if E is None:
        # print("Warning: Could not compute Essential Matrix.") # Optional print
        return None, None, None

    # Recover Rotation (R) and Translation (t)
    # recoverPose also expects points in shape (N, 2)
    _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, camera_matrix, mask=mask) # Pass mask from findEssentialMat

    if R is None or t is None:
         # print("Warning: Could not recover pose from Essential Matrix.") # Optional print
         return None, None, None

    return R, t, mask_pose # mask_pose indicates inliers consistent with the recovered pose

def process_video(video_path):
    """
    Reads frames from a video file, converts them to grayscale,
    detects ORB features, and processes them.


    Args:
        video_path (str): The path to the video file.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # --- Camera Intrinsics (Placeholder - Replace with actual values) ---
    # Assuming a hypothetical 640x480 camera with focal length ~500 pixels
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fx = 500 # Placeholder focal length in pixels
    fy = 500 # Placeholder focal length in pixels
    cx = width / 2
    cy = height / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    print(f"Using Camera Matrix:\n{K}")
    # --------------------------------------------------------------------

    # Initialize ORB detector and BFMatcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    frame_count = 0
    prev_kp = None
    prev_desc = None
    # prev_gray = None # Uncomment if drawing matches

    # Trajectory accumulation variables
    trajectory_points = [np.array([0.0, 0.0, 0.0])] # Start at origin
    current_R = np.eye(3) # Initial rotation is identity
    current_t = np.zeros((3, 1)) # Initial translation is zero
    # trajectory = np.zeros((600, 600, 3), dtype=np.uint8) # For drawing trajectory
    # current_pose_R = np.eye(3)
    # current_pose_t = np.zeros((3, 1))

    while True:
        ret, frame = cap.read()

        # If frame is not read correctly, break the loop
        if not ret:
            break

        # Detect features in the current frame
        gray_frame, kp, desc = detect_features(frame, orb)
        frame_count += 1

        # Match features with the previous frame
        good_matches, pts1, pts2 = match_features(prev_kp, prev_desc, kp, desc, bf)

        # Estimate pose if enough good matches are found
        R, t, pose_mask = None, None, None
        if len(good_matches) > 5: # Need at least 5 points for findEssentialMat
            R, t, pose_mask = estimate_pose(pts1, pts2, K)

        # --- Trajectory Accumulation ---
        if R is not None and t is not None:
            # Normalize translation vector (address scale ambiguity)
            t_norm = np.linalg.norm(t)
            if t_norm > 1e-6: # Avoid division by zero
                 t_normalized = t / t_norm
            else:
                 t_normalized = t # Keep zero vector if norm is too small

            # Update global pose
            # Note: t is relative translation in the *previous* camera frame's orientation
            # We transform it to the global frame using the *current* global rotation
            current_t = current_t + current_R @ t_normalized # Use normalized t
            current_R = R @ current_R # Update rotation (Note: R is rotation from prev to current)

            # Store the new position
            trajectory_points.append(current_t.flatten())

            current_pos_str = f"X:{current_t[0,0]:.2f}, Y:{current_t[1,0]:.2f}, Z:{current_t[2,0]:.2f}"
            print(f"Processing frame: {frame_count}, Keypoints: {len(kp)}, Matches: {len(good_matches)}, Pose Estimated: Yes, Pos: [{current_pos_str}]")

            # --- Draw Trajectory (Example - Needs refinement for scale/visualization) ---
            # x, y, z = current_t.flatten()
            # draw_x, draw_y = int(x*scale_factor) + 300, int(z*scale_factor) + 100 # Apply scale factor for drawing
            # cv2.circle(trajectory, (draw_x, draw_y), 1, (0, 255, 0), 1)
            # cv2.imshow('Trajectory', trajectory)
            # -----------------------------------------------------------------------
        else:
            print(f"Processing frame: {frame_count}, Keypoints: {len(kp)}, Matches: {len(good_matches)}, Pose Estimated: No")

        # Store current frame data for the next iteration (regardless of pose estimation success)
        prev_kp = kp
        prev_desc = desc

        # Optional: Draw matches for visualization
        # if frame_count > 1 and prev_gray is not None:
        #     img_matches = cv2.drawMatches(prev_gray, prev_kp, gray_frame, kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #     cv2.imshow('Matches', img_matches)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # prev_gray = gray_frame # Store current gray frame for next iteration's drawing

        # Optional: Draw keypoints for visualization
        # frame_with_keypoints = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)
        # cv2.imshow('Frame with Keypoints', frame_with_keypoints)
        # if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit display
        #     break
        # -----------------------------------------

    # Release the video capture object
    cap.release()
    # cv2.destroyAllWindows() # Uncomment if using cv2.imshow

    print(f"\nFinished processing video. Total frames: {frame_count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_video>")
        sys.exit(1)

    video_file = sys.argv[1]
    process_video(video_file)
