import cv2
import sys
import numpy as np # Needed for converting points later
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting to buffer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
import io # For plot buffer
import threading # For stopping mechanism
# import time # Uncomment if using time.sleep

# --- Helper Functions (can be methods if preferred) ---

def detect_features(frame, detector):
    """Detects keypoints and computes descriptors for a given frame."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Parameters can be tuned, e.g., nfeatures for ORB
    keypoints, descriptors = detector.detectAndCompute(gray_frame, None)
    return gray_frame, keypoints, descriptors

def match_features(kp1, desc1, kp2, desc2, matcher, ratio_thresh=0.75):
    """Matches features between two sets of descriptors using Lowe's ratio test."""
    good_matches = []
    pts1 = []
    pts2 = []
    # Ensure descriptors are not None and have enough features to match (k=2 requires at least 2)
    if desc1 is not None and desc2 is not None and len(desc1) >= 2 and len(desc2) >= 2:
        try:
            matches = matcher.knnMatch(desc1, desc2, k=2)
            # Apply Lowe's ratio test
            for m, n in matches:
                # Ensure m and n are valid matches before accessing distance
                if m and n and m.distance < ratio_thresh * n.distance:
                    # Ensure indices are within bounds before accessing keypoints
                    if m.queryIdx < len(kp1) and m.trainIdx < len(kp2):
                        good_matches.append(m)
                        pts1.append(kp1[m.queryIdx].pt)
                        pts2.append(kp2[m.trainIdx].pt)
                    else:
                        print(f"Warning: Match index out of bounds. queryIdx: {m.queryIdx}, trainIdx: {m.trainIdx}, kp1 len: {len(kp1)}, kp2 len: {len(kp2)}")

        except cv2.error as e:
            print(f"Error during knnMatch: {e}")
            # Handle cases where descriptors might be empty or have incompatible types/sizes
            pass
        except IndexError as e:
             print(f"Error accessing match objects (likely due to empty matches): {e}")
             pass


    # Convert points to float32 as required by findEssentialMat
    return good_matches, np.float32(pts1), np.float32(pts2)

def estimate_pose(pts1, pts2, camera_matrix, prob=0.999, threshold=1.0):
    """Estimates the relative camera pose from matched points."""
    # Need at least 5 points for findEssentialMat RANSAC
    if pts1 is None or pts2 is None or len(pts1) < 5 or len(pts2) < 5:
        return None, None, None

    try:
        # Calculate Essential Matrix using RANSAC
        E, mask = cv2.findEssentialMat(pts2, pts1, camera_matrix, method=cv2.RANSAC, prob=prob, threshold=threshold)

        if E is None or E.shape != (3, 3): # Check shape as findEssentialMat might return incorrect shapes sometimes
            # print("Warning: Could not compute Essential Matrix or incorrect shape.")
            return None, None, None

        # Recover Rotation (R) and Translation (t)
        points_recovered, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, camera_matrix, mask=mask)

        if points_recovered == 0 or R is None or t is None: # Check if pose recovery was successful
             # print("Warning: Could not recover pose from Essential Matrix.")
             return None, None, None

        return R, t, mask_pose # mask_pose indicates inliers consistent with the recovered pose
    except cv2.error as e:
        print(f"Error during pose estimation (findEssentialMat/recoverPose): {e}")
        return None, None, None


# --- VO Processor Class ---

class VO_Processor:
    def __init__(self, camera_matrix):
        self.K = camera_matrix
        self.orb = cv2.ORB_create(nfeatures=1000) # Increased features, consider making configurable
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.frame_count = 0
        self.prev_kp = None
        self.prev_desc = None

        self.trajectory_points = [np.array([0.0, 0.0, 0.0])]
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))

        # Plotting setup
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._plot_lock = threading.Lock() # Lock for thread-safe plot updates

        # State for Flask app
        self.current_frame = None
        self.current_plot_img = None
        self.current_status = {"frame": 0, "kp": 0, "matches": 0, "pose": "No", "x": 0.0, "y": 0.0, "z": 0.0}
        self.is_running = False
        self._stop_event = threading.Event()
        self._update_plot() # Initialize plot image buffer

    def _update_plot(self):
        """Renders the current trajectory plot to an image buffer (thread-safe)."""
        with self._plot_lock:
            if len(self.trajectory_points) > 1:
                trajectory_array = np.array(self.trajectory_points)
                x = trajectory_array[:, 0]
                y = trajectory_array[:, 1]
                z = trajectory_array[:, 2]

                self.ax.cla()
                self.ax.plot(x, y, z, marker='.', linestyle='-', markersize=2)
                self.ax.set_xlabel("X")
                self.ax.set_ylabel("Y")
                self.ax.set_zlabel("Z")
                self.ax.set_title("Estimated Trajectory (3D)")
                self.ax.grid(True)
                # Auto-scaling is usually best for 3D buffer rendering

                buf = io.BytesIO()
                self.fig.savefig(buf, format='png', dpi=self.fig.dpi)
                buf.seek(0)
                plot_img_np = np.frombuffer(buf.getvalue(), dtype=np.uint8)
                buf.close()
                self.current_plot_img = cv2.imdecode(plot_img_np, cv2.IMREAD_COLOR)
            else:
                # Create a placeholder if no points yet
                placeholder = np.zeros((int(self.fig.dpi*6), int(self.fig.dpi*6), 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for data...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.current_plot_img = placeholder


    def process_frame(self, frame):
        """Processes a single frame for VO."""
        if frame is None:
            print("Error: Received None frame in process_frame")
            return False
        self.current_frame = frame.copy() # Store original frame for video feed

        gray_frame, kp, desc = detect_features(frame, self.orb)
        self.frame_count += 1

        good_matches, pts1, pts2 = match_features(self.prev_kp, self.prev_desc, kp, desc, self.bf)

        R, t, pose_mask = None, None, None
        pose_estimated = False
        if len(good_matches) > 5:
            R, t, pose_mask = estimate_pose(pts1, pts2, self.K)

        if R is not None and t is not None:
            pose_estimated = True
            t_norm = np.linalg.norm(t)
            t_normalized = t / t_norm if t_norm > 1e-6 else t

            # Update global pose
            self.current_t = self.current_t + self.current_R @ t_normalized
            self.current_R = R @ self.current_R
            self.trajectory_points.append(self.current_t.flatten())

            self._update_plot() # Update plot image buffer

        # Update status dictionary
        current_pos = self.current_t.flatten()
        self.current_status = {
            "frame": self.frame_count,
            "kp": len(kp) if kp is not None else 0,
            "matches": len(good_matches),
            "pose": "Yes" if pose_estimated else "No",
            "x": float(current_pos[0]),
            "y": float(current_pos[1]),
            "z": float(current_pos[2])
        }

        # Update previous frame data
        self.prev_kp = kp
        self.prev_desc = desc

        return pose_estimated

    def run(self, video_path):
        """Main processing loop to be run in a thread."""
        self.is_running = True
        self._stop_event.clear()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file in processor: {video_path}")
            self.is_running = False
            return

        # Reset state for new run
        self.frame_count = 0
        self.prev_kp = None
        self.prev_desc = None
        self.trajectory_points = [np.array([0.0, 0.0, 0.0])]
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))
        self._update_plot() # Initialize plot

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("End of video or cannot read frame.")
                break

            self.process_frame(frame)

            # Small sleep to prevent CPU hogging if processing is very fast
            # time.sleep(0.01) # Requires importing time

        cap.release()
        plt.close(self.fig) # Close the figure when done
        self.is_running = False
        print("VO Processing stopped.")

    def stop(self):
        """Signals the processing loop to stop."""
        self._stop_event.set()

    def get_current_frame(self):
        """Returns the latest processed frame."""
        # Return a default blank image if current_frame is None
        if self.current_frame is None:
             # You might want to get dimensions from K or have defaults
             h, w = 480, 640 # Default dimensions
             blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
             cv2.putText(blank_frame, "Waiting for video...", (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
             return blank_frame
        return self.current_frame

    def get_plot_image(self):
        """Returns the latest rendered plot image."""
        # Return placeholder if plot image is None
        if self.current_plot_img is None:
             self._update_plot() # Attempt to generate plot
        # If still None after update (e.g., error), return a placeholder
        if self.current_plot_img is None:
            h, w = int(self.fig.dpi*6), int(self.fig.dpi*6) # Placeholder size
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Plot unavailable", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return placeholder
        return self.current_plot_img


    def get_status(self):
        """Returns the current status dictionary."""
        return self.current_status
