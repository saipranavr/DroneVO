import cv2
import sys

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

    # Initialize ORB detector
    orb = cv2.ORB_create()

    frame_count = 0
    prev_keypoints = None
    prev_descriptors = None

    while True:
        ret, frame = cap.read()

        # If frame is not read correctly, break the loop
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray_frame, None)

        frame_count += 1

        # --- Placeholder for further processing (e.g., feature matching) ---
        print(f"Processing frame: {frame_count}, Keypoints detected: {len(keypoints)}")

        # Store current keypoints and descriptors for the next iteration
        # (In a full VO system, you'd match features between prev_descriptors and descriptors)
        prev_keypoints = keypoints
        prev_descriptors = descriptors

        # Optional: Draw keypoints for visualization
        # frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0,255,0), flags=0)
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
