import cv2
import sys
import numpy as np # Needed for converting points later
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting to buffer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
import io # For plot buffer

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
    # prev_draw_x, prev_draw_y = -1, -1 # No longer needed for drawing on frame

    # --- Matplotlib Setup for 3D Plotting to Buffer ---
    fig = plt.figure(figsize=(6, 6)) # Adjust figsize as needed
    ax = fig.add_subplot(111, projection='3d')
    # ---------------------------------------------

    # Define target display size (adjust as needed)
    display_height = 480

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

        # Initialize plot image for this iteration
        plot_img = None

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
            current_pos = current_t.flatten()
            trajectory_points.append(current_pos)

            current_pos_str = f"X:{current_pos[0]:.2f}, Y:{current_pos[1]:.2f}, Z:{current_pos[2]:.2f}"
            print(f"Processing frame: {frame_count}, Keypoints: {len(kp)}, Matches: {len(good_matches)}, Pose Estimated: Yes, Pos: [{current_pos_str}]")

            # --- Render Matplotlib Plot to Image Buffer ---
            plot_img = None
            # --- Render Matplotlib 3D Plot to Image Buffer ---
            plot_img = None
            if len(trajectory_points) > 1:
                trajectory_array = np.array(trajectory_points)
                x_coords = trajectory_array[:, 0]
                y_coords = trajectory_array[:, 1] # Use Y coordinate
                z_coords = trajectory_array[:, 2]

                ax.cla() # Clear previous plot
                ax.plot(x_coords, y_coords, z_coords, marker='.', linestyle='-', markersize=2, label='Trajectory') # Plot X, Y, Z
                ax.set_xlabel("X")
                ax.set_ylabel("Y") # Set Y label
                ax.set_zlabel("Z") # Set Z label
                ax.set_title("Estimated Camera Trajectory (3D)")
                # ax.legend(loc='upper left') # Optional legend
                ax.grid(True)
                # Setting equal aspect ratio in 3D can be tricky, might need manual limits
                # ax.set_aspect('equal') # May not work well in 3D
                # Auto-scaling usually works better for 3D

                # Use Agg backend to draw plot to buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=fig.dpi)
                buf.seek(0)
                plot_img_np = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                buf.close()
                plot_img = cv2.imdecode(plot_img_np, cv2.IMREAD_COLOR)
            # ------------------------------------------

        else:
             print(f"Processing frame: {frame_count}, Keypoints: {len(kp)}, Matches: {len(good_matches)}, Pose Estimated: No")
             # Ensure plot_img remains None if pose estimation failed
             plot_img = None

        # --- Combine Video Frame and Plot Image ---
        # Resize frame
        frame_aspect_ratio = width / height
        frame_display_width = int(display_height * frame_aspect_ratio)
        resized_frame = cv2.resize(frame, (frame_display_width, display_height))

        # Create placeholder if plot not ready, else resize plot
        if plot_img is None:
            # Use a black placeholder if plot isn't generated yet (e.g., first frame)
            plot_display_width = display_height # Assume square plot for placeholder
            resized_plot = np.zeros((display_height, plot_display_width, 3), dtype=np.uint8)
        else:
            plot_height, plot_width, _ = plot_img.shape
            plot_aspect_ratio = plot_width / plot_height
            plot_display_width = int(display_height * plot_aspect_ratio)
            resized_plot = cv2.resize(plot_img, (plot_display_width, display_height))

        # Create combined image canvas
        total_width = frame_display_width + plot_display_width
        combined_display = np.zeros((display_height, total_width, 3), dtype=np.uint8)

        # Place resized frame and plot side-by-side
        combined_display[0:display_height, 0:frame_display_width] = resized_frame
        combined_display[0:display_height, frame_display_width:total_width] = resized_plot

        # Display the combined image
        cv2.imshow('Visual Odometry Dashboard', combined_display)
        # ------------------------------------------

        # Store current frame data for the next iteration (regardless of pose estimation success)
        prev_kp = kp
        prev_desc = desc

        # Exit loop if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'): # Adjust delay (e.g., 1 for max speed, 30 for ~30fps)
             break
        # -----------------------------------------

    # Release the video capture object and destroy windows
    cap.release()
    cv2.destroyAllWindows()
    plt.close(fig) # Close the matplotlib figure
    print(f"\nFinished processing video. Total frames: {frame_count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_video>")
        sys.exit(1)

    video_file = sys.argv[1]
    process_video(video_file)
