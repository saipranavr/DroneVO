import cv2
import sys

def process_video(video_path):
    """
    Reads frames from a video file, converts them to grayscale,
    and processes them (currently just prints frame count).

    Args:
        video_path (str): The path to the video file.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()

        # If frame is not read correctly, break the loop
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_count += 1

        # --- Placeholder for further processing ---
        # For now, just print the frame number being processed
        # In future steps, this is where VO logic would go.
        print(f"Processing frame: {frame_count}")
        # You could display the frame using cv2.imshow if needed for debugging:
        # cv2.imshow('Grayscale Frame', gray_frame)
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
