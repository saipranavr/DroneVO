import os
import sys # Add sys import
print(f"--- Debug: Running with Python executable: {sys.executable}")
print(f"--- Debug: sys.path:")
for p in sys.path:
    print(f"    {p}")
print("--- Debug: Attempting to import cv2 ---")
try:
    import cv2
    print("--- Debug: cv2 imported successfully ---")
except ImportError as e:
    print(f"--- Debug: Failed to import cv2: {e} ---")
    # Exit if cv2 cannot be imported, as the app depends on it
    sys.exit(f"Critical Error: OpenCV (cv2) not found in environment {sys.executable}")

import threading
import time
# import cv2 # Already imported above
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import logging # Add logging
import io # For plot buffer

# Assuming visual_odometry.py is in the same directory
from visual_odometry import VO_Processor

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'} # Add video extensions as needed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__, template_folder='templates') # Explicitly set template folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key' # Needed for flash messages, change in production

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global Variables ---
# Placeholder Camera Matrix (Ideally load from calibration or config)
# Using default values assuming a 640x480 frame, modify if needed
fx = 500; fy = 500; cx = 320; cy = 240
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

vo_processor = VO_Processor(camera_matrix)
vo_thread = None
vo_lock = threading.Lock() # Lock for accessing shared vo_processor and vo_thread

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    # Check if index.html exists, otherwise return a simple message
    template_path = os.path.join(app.template_folder, 'index.html')
    logging.info(f"Looking for template at: {template_path}")
    if not os.path.exists(template_path):
        logging.error("index.html template not found.")
        return "Error: HTML template 'templates/index.html' not found."
    return render_template('index.html')

@app.route('/start_processing', methods=['POST'])
def start_processing():
    """Handles video upload and starts VO processing in a background thread."""
    global vo_thread, vo_processor
    logging.info("Received request to start processing.")

    if 'videoFile' not in request.files:
        logging.warning("No file part in request.")
        return jsonify({"error": "No file part"}), 400
    file = request.files['videoFile']
    if file.filename == '':
        logging.warning("No selected file.")
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logging.info(f"Attempting to save uploaded file to: {video_path}")
        try:
            file.save(video_path)
            logging.info(f"File saved successfully: {video_path}")
        except Exception as e:
             logging.error(f"Failed to save file: {e}")
             return jsonify({"error": f"Failed to save file: {e}"}), 500

        with vo_lock: # Acquire lock to modify shared resources
            if vo_processor.is_running:
                logging.info("Stopping previous processing run...")
                vo_processor.stop() # Stop previous run if any
                if vo_thread and vo_thread.is_alive():
                    vo_thread.join(timeout=2.0) # Wait a bit longer for thread to finish

            # Re-initialize processor for new run (clears old state)
            logging.info("Initializing new VO Processor...")
            vo_processor = VO_Processor(camera_matrix) # Pass the camera matrix

            # Start processing in a new thread
            logging.info(f"Starting processing thread for {video_path}")
            vo_thread = threading.Thread(target=vo_processor.run, args=(video_path,), daemon=True)
            vo_thread.start()

        return jsonify({"message": "Processing started", "filename": filename}), 200
    else:
        logging.warning(f"File type not allowed: {file.filename}")
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    """Stops the background VO processing thread."""
    global vo_thread, vo_processor
    logging.info("Received request to stop processing.")
    stopped = False
    with vo_lock:
        if vo_processor.is_running:
            logging.info("Signaling VO processor to stop...")
            vo_processor.stop()
            stopped = True
        else:
            logging.info("Processing not running.")
            return jsonify({"message": "Processing not running"}), 200

    # Wait for thread outside the lock to avoid holding it too long
    if stopped and vo_thread and vo_thread.is_alive():
        logging.info("Waiting for VO thread to join...")
        vo_thread.join(timeout=2.0) # Wait a bit longer
        if not vo_thread.is_alive():
            logging.info("VO thread stopped successfully.")
            return jsonify({"message": "Processing stopped"}), 200
        else:
            logging.warning("VO thread did not stop in time.")
            return jsonify({"message": "Stop signal sent, but thread might still be running"}), 202
    elif stopped: # Stop was called, but thread might have already finished
         logging.info("Stop signal sent, thread already finished or joined.")
         return jsonify({"message": "Processing stopped"}), 200
    else: # Should not happen if logic above is correct
         return jsonify({"message": "Processing not running"}), 200


def generate_video_feed():
    """Generator function to stream video frames."""
    logging.info("Video feed connection established.")
    last_frame_time = time.time()
    while True:
        with vo_lock:
            frame = vo_processor.get_current_frame() # Access shared resource under lock

        if frame is None:
            # Send placeholder if no frame available yet
            h, w = 480, 640 # Default dimensions
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for video...", (50, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        try:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.error("Error encoding video frame.")
                time.sleep(0.1) # Avoid busy-looping on error
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logging.error(f"Error generating/yielding video frame: {e}")
            time.sleep(0.1)
            pass # Avoid crashing the feed if one frame fails

        # Control frame rate
        time_elapsed = time.time() - last_frame_time
        sleep_time = max(0, (1.0/30.0) - time_elapsed) # Aim for ~30 FPS
        time.sleep(sleep_time)
        last_frame_time = time.time()


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_plot_feed():
    """Generator function to stream plot images."""
    logging.info("Plot feed connection established.")
    while True:
        with vo_lock:
            plot_img = vo_processor.get_plot_image() # Access shared resource under lock

        if plot_img is None:
             logging.warning("Plot image is None, waiting...")
             time.sleep(0.2) # Wait longer if plot isn't ready
             continue

        try:
            ret, buffer = cv2.imencode('.png', plot_img) # Use PNG for plots
            if not ret:
                logging.error("Error encoding plot image.")
                time.sleep(0.1)
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/png\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logging.error(f"Error generating/yielding plot frame: {e}")
            time.sleep(0.1)
            pass
        time.sleep(0.1) # Update plot less frequently than video

@app.route('/trajectory_plot')
def trajectory_plot():
    """Trajectory plot streaming route."""
    return Response(generate_plot_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Returns the current VO status as JSON."""
    with vo_lock:
        current_status = vo_processor.get_status()
        current_status['is_running'] = vo_processor.is_running # Add running state
    return jsonify(current_status)

# Serve uploaded files (optional, for debugging or direct access)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # Note: Using Flask's development server is not recommended for production.
    # Consider using a production-ready server like Gunicorn or Waitress.
    logging.info("Starting Flask server on http://0.0.0.0:5000")
    # Use threaded=True to handle background tasks and multiple requests
    # Use use_reloader=False to prevent issues with background threads in debug mode
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5001, use_reloader=False)
