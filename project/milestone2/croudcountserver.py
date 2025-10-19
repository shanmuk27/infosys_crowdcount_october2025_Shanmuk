from flask import Flask, render_template, redirect, url_for, request, session, make_response, jsonify, Response
import datetime
import pyrebase
import jwt
import os
from ultralytics import YOLO
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import threading
import time

config = {
    'apiKey': "AIzaSyAHT27Lik4POA67GsBkeVUHWvWYeUqysi4",
    'authDomain': "crowd-count-3e92c.firebaseapp.com",
    'databaseURL': "https://crowd-count-3e92c-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "crowd-count-3e92c",
    'storageBucket': "crowd-count-3e92c.firebasestorage.app",
    'messagingSenderId': "605062371417",
    'appId': "1:605062371417:web:6e259e609153b20640f948",
    'measurementId': "G-GXK3GQKMHG"
}

firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
db = firebase.database()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")
person_class_id = 0
confidence_threshold = 0.4
iou_threshold = 0.7
sel = 0

# --- GLOBAL CONTROL VARIABLES FOR THREADING ---
IS_PROCESSING_VIDEO = False
IS_PROCESSING_CAMERA = False
# ----------------------------------------------

def process_frame_results(results_generator):
    if not results_generator:
        return None
        
    try:
        result = next(iter(results_generator))
    except StopIteration:
        return None
    
    if result is None or result.orig_img is None:
        return None
            
    frame = result.orig_img
    
    person_id_counter = 1
    
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        label = str(person_id_counter)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = x1 + (x2 - x1) // 2 - text_size[0] // 2
        text_y = y1 - 10
        padding = 3 
        
        cv2.rectangle(frame, (text_x - padding, text_y - text_size[1] - padding), 
                      (text_x + text_size[0] + padding, text_y + padding), 
                      color, -1)
        
        cv2.putText(frame, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        person_id_counter += 1

    return frame

# --- NEW THREADING FUNCTIONS ---

def process_video_in_background(video_path, area_name):
    global IS_PROCESSING_VIDEO
    IS_PROCESSING_VIDEO = True

    try:
        results_generator = model.predict(
            source=video_path,
            classes=[person_class_id], 
            conf=confidence_threshold,
            iou=iou_threshold,
            stream=True,
            verbose=False,
            # tracker='bytetrack.yaml' # Keep tracker disabled for simple streaming
        )
        
        # NOTE: cv2.imshow requires a loop and cv2.waitKey on the main thread for optimal performance.
        # Running it in a thread might cause issues depending on your OS/GUI backend, but we proceed.
        for result in results_generator:
            if not IS_PROCESSING_VIDEO:
                break
            
            processed_frame = process_frame_results([result])
            if processed_frame is not None:
                cv2.imshow(f"Video Analysis: {area_name}", processed_frame)
                
                # Check for 'q' key press to stop processing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Simple delay to prevent overwhelming the display loop
            time.sleep(0.01) 
            
    except Exception as e:
        print(f"Error during video processing: {e}")
        
    finally:
        IS_PROCESSING_VIDEO = False
        cv2.destroyAllWindows()
        print(f"Finished processing video: {video_path} for {area_name}")

def process_camera_in_background(area_name):
    global IS_PROCESSING_CAMERA
    IS_PROCESSING_CAMERA = True
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        IS_PROCESSING_CAMERA = False
        return
        
    try:
        while IS_PROCESSING_CAMERA:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break
                
            results = model.predict(
                source=frame,
                classes=[person_class_id], 
                conf=confidence_threshold,
                iou=iou_threshold,
                stream=False,
                verbose=False,
            )
            
            processed_frame = process_frame_results(results)
            
            if processed_frame is not None:
                cv2.imshow(f"Live Camera Analysis: {area_name}", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error during camera processing: {e}")
        
    finally:
        IS_PROCESSING_CAMERA = False
        cap.release()
        cv2.destroyAllWindows()
        print(f"Finished processing camera for {area_name}")

# ----------------------------------------------

app = Flask(__name__)
app.secret_key = 'a-very-secret-key'
JWT_SECRET = 'another-very-secret-key'
JWT_ALGORITHM = 'HS256'

def get_user_data(uid, id_token):
    return db.child("user").child(uid).get(id_token).val()

def generate_server_jwt(uid, email, username):
    current_time = datetime.datetime.utcnow()
    payload = {
        "uid": uid,
        "email": email,
        "username": username,
        "iat": current_time,
        "exp": current_time + datetime.timedelta(hours=1)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_server_jwt(token):
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def check_auth(request):
    jwt_token = request.cookies.get('server_jwt')
    return verify_server_jwt(jwt_token)

@app.route('/')
def root():
    return redirect(url_for('signin'))

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/login', methods=['POST'])
def login():
    try:
        email = request.form['email']
        password = request.form['password']
        
        user = auth.sign_in_with_email_and_password(email, password)
        uid = user['localId']
        id_token = user['idToken']

        user_data = get_user_data(uid, id_token)
        username = user_data.get('user_name', 'User')

        server_jwt = generate_server_jwt(uid, email, username)

        session['user'] = {'uid': uid, 'idToken': id_token, 'jwt': server_jwt}

        response = make_response(redirect(url_for('index')))
        response.set_cookie('server_jwt', server_jwt, httponly=True, samesite='Lax')
        return response

    except Exception as e:
        print(e)
        return render_template('signin.html', Error="Invalid username/password!!")

@app.route('/register', methods=['POST'])
def register():
    try:
        email = request.form['email']
        password = request.form['password']
        username = request.form['username']

        user = auth.create_user_with_email_and_password(email, password)
        uid = user['localId']
        id_token = user['idToken']

        data = {"user_name": username}
        db.child("user").child(uid).set(data, id_token)

        return redirect(url_for('signin'))

    except Exception as e:
        print(e)
        error = "Could not register. The email might already be in use."
        return render_template('signin.html', Error=error, mode='register')

@app.route('/index')
def index():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    
    username = decoded.get('username', 'User')
    return render_template('index.html', username=username)

@app.route('/profile')
def profile():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))

    username = decoded.get('username', 'User')
    email = decoded.get('email', '')
    return render_template('profile.html', username=username, email=email)

@app.route('/home')
def home():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    
    return render_template('home.html')

@app.route('/vid_analy')
def vid_analy():
    return render_template("vid_analy.html")

@app.route('/cam_manage', methods=['GET'])
def cam_manage_page():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    return render_template("cam_manage.html")

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global IS_PROCESSING_VIDEO, IS_PROCESSING_CAMERA
    
    if IS_PROCESSING_VIDEO or IS_PROCESSING_CAMERA:
        return jsonify({"message": "A video or camera process is already running."}), 409
        
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    area_name = request.form.get("areaName", "Unknown")
    
    filename = secure_filename(video_file.filename)
    full_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(full_path)
    
    if not os.path.exists(full_path):
        return jsonify({"error": f"Failed to save video at: {full_path}"}), 500

    # START PROCESSING IN A BACKGROUND THREAD
    threading.Thread(target=process_video_in_background, args=(full_path, area_name)).start()
    
    # Return immediately to the client so the web app doesn't block
    return jsonify({"message": f"Video processing started on server for {area_name}. Check the display window."}), 200

@app.route('/upload_cam', methods=['POST'])
def upload_cam():
    global IS_PROCESSING_CAMERA, IS_PROCESSING_VIDEO
    
    if IS_PROCESSING_CAMERA or IS_PROCESSING_VIDEO:
        return jsonify({"message": "A camera or video process is already running."}), 409

    area_name = request.form.get("areaName", "Unknown")
    
    # START PROCESSING IN A BACKGROUND THREAD
    threading.Thread(target=process_camera_in_background, args=(area_name,)).start()

    # Return immediately to the client so the web app doesn't block
    return jsonify({"message": f"Camera stream initiated on server for {area_name}. Check the display window."}), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global IS_PROCESSING_VIDEO, IS_PROCESSING_CAMERA
    
    if not IS_PROCESSING_VIDEO and not IS_PROCESSING_CAMERA:
        return jsonify({"message": "No stream is currently active."}), 200
        
    # Set the global flags to False to break the loops in the background threads
    IS_PROCESSING_VIDEO = False
    IS_PROCESSING_CAMERA = False
    
    # Give the thread a moment to shut down and close the windows
    time.sleep(1) 
    cv2.destroyAllWindows() 
    return jsonify({"message": "Video/Camera processing stopped on server."}), 200


@app.route('/signout')
def signout():
    global IS_PROCESSING_VIDEO, IS_PROCESSING_CAMERA
    
    # Ensure any running processes are terminated on signout
    IS_PROCESSING_VIDEO = False
    IS_PROCESSING_CAMERA = False
    cv2.destroyAllWindows()
    
    session.pop('user', None)
    response = make_response(redirect(url_for('signin')))
    response.set_cookie('server_jwt', '', expires=0)
    return response

if __name__ == '__main__':
    # Running with 'threaded=True' is essential for this approach
    app.run(debug=True, threaded=True)