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
import queue 

STREAM_SOURCES = {} 
THREAD_RESULTS = {}
ACTIVE_THREADS = {}
LIVE_COUNTS = {} 

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

def process_frame_results(results_generator):
    if not results_generator:
        return None, 0
        
    try:
        result = next(iter(results_generator))
    except StopIteration:
        return None, 0
    
    if result is None or result.orig_img is None:
        return None, 0
            
    frame = result.orig_img
    
    person_count = len(result.boxes)
    
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

    return frame, person_count

def video_stream_generator(source, area_name):
    global LIVE_COUNTS, STREAM_SOURCES
    
    if source == "0":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open source for {area_name}")
        LIVE_COUNTS[area_name] = 0
        return

    frame_count = 0
    final_count = 0
    
    try:
        while True:
            if area_name not in STREAM_SOURCES:
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(
                source=frame,
                classes=[person_class_id], 
                conf=confidence_threshold,
                iou=iou_threshold,
                stream=False, 
                verbose=False,
            )

            processed_frame, current_count = process_frame_results(results)

            if processed_frame is not None:
                LIVE_COUNTS[area_name] = current_count 
                
                ret, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                final_count = current_count 
                frame_count += 1
                
    except Exception as e:
        print(f"Streaming error for {area_name}: {e}")
        THREAD_RESULTS[area_name] = f"Error: {e}"
        
    finally:
        cap.release()
        
        if area_name in STREAM_SOURCES:
            del STREAM_SOURCES[area_name]
            
        if area_name in LIVE_COUNTS:
            del LIVE_COUNTS[area_name]
            
        THREAD_RESULTS[area_name] = f"Stream ended. Total frames processed: {frame_count}. Last count: {final_count}"
        print(f"Finished processing and stored result for {area_name}: {THREAD_RESULTS[area_name]}")


app = Flask(__name__)
app.secret_key = 'a-very-secret-key'
JWT_SECRET = 'another-very-secret-key'
JWT_ALGORITHM = 'HS256'

def get_user_data(uid, id_token):
    return db.child("user").child(uid).get(id_token).val()

def generate_server_jwt(uid, email, username):
    current_time = datetime.datetime.now(datetime.UTC) 
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
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
        
    camera_data = [
        {'areaName': name, 'sourceType': 'Webcam' if STREAM_SOURCES[name] == '0' else 'Video File'}
        for name in STREAM_SOURCES.keys()
    ]

    return render_template("vid_analy.html", cameras=camera_data)


@app.route('/cam_manage', methods=['GET'])
def cam_manage_page():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    return render_template("cam_manage.html")

@app.route('/video_feed/<area_name>')
def video_feed(area_name):
    source = STREAM_SOURCES.get(area_name)
    if not source:
        return "No active stream for this area.", 404

    return Response(
        video_stream_generator(source, area_name),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/get_count/<area_name>', methods=['GET'])
def get_count(area_name):
    current_count = LIVE_COUNTS.get(area_name, "N/A")
    is_active = area_name in STREAM_SOURCES
    
    return jsonify({
        "areaName": area_name,
        "count": current_count,
        "isActive": is_active
    })


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    area_name = request.form.get("areaName", "Unknown")
    
    if area_name in STREAM_SOURCES:
        return jsonify({"message": f"A stream is already active for {area_name}. Stop it first."}), 409
        
    filename = secure_filename(video_file.filename)
    full_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(full_path)
    
    if not os.path.exists(full_path):
        return jsonify({"error": f"Failed to save video at: {full_path}"}), 500

    STREAM_SOURCES[area_name] = full_path
    LIVE_COUNTS[area_name] = 0
    
    return jsonify({"message": f"Video source saved for {area_name}. Stream will start on dashboard view."}), 200

@app.route('/upload_cam', methods=['POST'])
def upload_cam():
    area_name = request.form.get("areaName", "Unknown")
    
    if area_name in STREAM_SOURCES:
        return jsonify({"message": f"A stream is already active for {area_name}. Stop it first."}), 409

    STREAM_SOURCES[area_name] = "0"
    LIVE_COUNTS[area_name] = 0

    return jsonify({"message": f"Camera source saved for {area_name}. Stream will start on dashboard view."}), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    area_name = request.json.get("areaName")
    
    if area_name in STREAM_SOURCES:
        if area_name in STREAM_SOURCES:
            del STREAM_SOURCES[area_name]
        
        message = f"Stream stop signal sent for {area_name}."
    else:
        message = "No active stream found to stop."
    
    return jsonify({"message": message}), 200


@app.route('/signout')
def signout():
    global STREAM_SOURCES, LIVE_COUNTS
    STREAM_SOURCES = {}
    LIVE_COUNTS = {}
    cv2.destroyAllWindows()
    
    session.pop('user', None)
    response = make_response(redirect(url_for('signin')))
    response.set_cookie('server_jwt', '', expires=0)
    return response

if __name__ == '__main__':
    app.run(debug=True, threaded=True)