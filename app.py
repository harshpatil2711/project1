from flask import Flask, jsonify, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
import uuid

app = Flask(__name__)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load filters for male and female
FILTERS_DIR_MALE = "static/filters/male"
filters_male = {f: cv2.imread(os.path.join(FILTERS_DIR_MALE, f), cv2.IMREAD_UNCHANGED) for f in os.listdir(FILTERS_DIR_MALE) if f.endswith('.png')}

FILTERS_DIR_FEMALE = "static/filters/female"
filters_female = {f: cv2.imread(os.path.join(FILTERS_DIR_FEMALE, f), cv2.IMREAD_UNCHANGED) 
                for f in os.listdir(FILTERS_DIR_FEMALE) if f.endswith('.png')}
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Please check if the camera is connected.")

# Global variables
selected_filter = None
current_gender = "male"  # Default gender
image_counter = 0

def overlay_image(background, overlay, x, y, width, height):
    """ Overlays an image (with alpha channel) on the background at (x, y). """
    overlay = cv2.resize(overlay, (width, height))
    h_bg, w_bg, _ = background.shape
    x = max(0, min(x, w_bg - width))
    y = max(0, min(y, h_bg - height))
    h, w, _ = overlay.shape
    overlay = overlay[:h, :w]

    if overlay.shape[2] == 4:  # Ensure alpha channel exists
        alpha_mask = overlay[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha_mask * overlay[:, :, c] + (1 - alpha_mask) * background[y:y+h, x:x+w, c]
            )
    return background

def generate_frames():
    global selected_filter, current_gender
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            print("Face detected!")  # Debugging
            
            # Get the appropriate filter image
            filter_img = None
            if current_gender == "male" and selected_filter in filters_male:
                filter_img = filters_male[selected_filter]
            elif current_gender == "female" and selected_filter in filters_female:
                filter_img = filters_female[selected_filter]
            
            # Apply the filter if available
            if filter_img is not None:
                for face_landmarks in results.multi_face_landmarks:
                    if selected_filter == "spect1rem.png":
                        # Special handling for glasses
                        left_eye_landmarks = [face_landmarks.landmark[i] for i in range(33, 133)]
                        right_eye_landmarks = [face_landmarks.landmark[i] for i in range(263, 362)]

                        # Calculate bounding box for left eye
                        left_eye_x = [int(lm.x * frame.shape[1]) for lm in left_eye_landmarks]
                        left_eye_y = [int(lm.y * frame.shape[0]) for lm in left_eye_landmarks]
                        left_eye_x1, left_eye_x2 = min(left_eye_x), max(left_eye_x)
                        left_eye_y1, left_eye_y2 = min(left_eye_y), max(left_eye_y)
                        left_eye_width = left_eye_x2 - left_eye_x1
                        left_eye_height = left_eye_y2 - left_eye_y1

                        # Calculate bounding box for right eye
                        right_eye_x = [int(lm.x * frame.shape[1]) for lm in right_eye_landmarks]
                        right_eye_y = [int(lm.y * frame.shape[0]) for lm in right_eye_landmarks]
                        right_eye_x1, right_eye_x2 = min(right_eye_x), max(right_eye_x)
                        right_eye_y1, right_eye_y2 = min(right_eye_y), max(right_eye_y)
                        right_eye_width = right_eye_x2 - right_eye_x1
                        right_eye_height = right_eye_y2 - right_eye_y1

                        # Overlay filter on eyes
                        frame = overlay_image(frame, filter_img, left_eye_x1, left_eye_y1, left_eye_width, left_eye_height)
                        frame = overlay_image(frame, filter_img, right_eye_x1, right_eye_y1, right_eye_width, right_eye_height)
                    else:
                        # Standard face positioning
                        x1 = int(face_landmarks.landmark[10].x * frame.shape[1])
                        y1 = int(face_landmarks.landmark[10].y * frame.shape[0])
                        x2 = int(face_landmarks.landmark[152].x * frame.shape[1])
                        y2 = int(face_landmarks.landmark[152].y * frame.shape[0])
                        x = (x1 + x2) // 2 - 80
                        y = (y1 + y2) // 2 - 150
                        frame = overlay_image(frame, filter_img, x, y, width=180, height=100)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    """ Render main page with filter list """
    global current_gender
    filters = filters_male if current_gender == "male" else filters_female
    return render_template('index.html', filters=list(filters.keys()), current_gender=current_gender)

@app.route('/set_filter/<gender>/<filter_name>')
def set_filter(gender, filter_name):
    global selected_filter, current_gender
    current_gender = gender  # Sync the gender state
    
    filters_dir = os.path.join("static/filters", gender)
    if os.path.exists(os.path.join(filters_dir, filter_name)):
        selected_filter = filter_name
        print(f"Filter set: {selected_filter} (Gender: {gender})")
        return "Filter set successfully", 200
    else:
        print(f"Filter not found: {filter_name} in {gender} folder")
        return "Filter not found", 404
@app.route('/get_filters/<gender>')
def get_filters(gender):
    """ Return a list of filters for the specified gender """
    filters_dir = os.path.join("static/filters", gender)
    print(f"Fetching filters from: {filters_dir}")  # Debugging
    if os.path.exists(filters_dir):
        filters = [f for f in os.listdir(filters_dir) if f.endswith('.png')]
        
        print(f"Filters found: {filters}")  # Debugging
        return jsonify(filters)
    else:
        print(f"Folder not found: {filters_dir}")  # Debugging
        return jsonify([]), 404

@app.route('/video_feed')
def video_feed():
    """ Video streaming route """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/capture_photo')
def capture_photo():
    global image_counter
    
    try:
        # Create gallery directory if needed
        os.makedirs('static/gallery', exist_ok=True)
        
        # Get current frame
        ret, frame = cap.read()
        if not ret:
            return jsonify({'status': 'error', 'message': 'Camera error'}), 500
        
        # Apply current filter
        if selected_filter:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                filter_img = filters_female.get(selected_filter) if current_gender == "female" else filters_male.get(selected_filter)
                
                if filter_img is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        # Apply filter positioning
                        x1 = int(face_landmarks.landmark[10].x * frame.shape[1])
                        y1 = int(face_landmarks.landmark[10].y * frame.shape[0])
                        x2 = int(face_landmarks.landmark[152].x * frame.shape[1])
                        y2 = int(face_landmarks.landmark[152].y * frame.shape[0])
                        x = (x1 + x2) // 2 - 80
                        y = (y1 + y2) // 2 - 150
                        frame = overlay_image(frame, filter_img, x, y, 180, 100)

        # Save with sequential naming
        image_counter += 1
        filename = f"image{image_counter}.jpg"
        filepath = os.path.join('static/gallery', filename)
        cv2.imwrite(filepath, frame)
        
        # Return both status and the image path for preview
        return jsonify({
            'status': 'success',
            'image_url': f'/static/gallery/{filename}?t={time.time()}'
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)