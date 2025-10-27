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

# NEW Global State for persistent processing
FRAME_QUEUES = {} 
ACTIVE_THREADS = {} 

STREAM_SOURCES = {} 
THREAD_RESULTS = {}
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

def continuous_video_analysis(source, area_name):
    global LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS, STREAM_SOURCES
    
    cap = cv2.VideoCapture(int(source) if source == "0" else source)

    if not cap.isOpened():
        print(f"Error: Cannot open source for {area_name}")
        if area_name in STREAM_SOURCES: del STREAM_SOURCES[area_name]
        if area_name in LIVE_COUNTS: del LIVE_COUNTS[area_name]
        if area_name in FRAME_QUEUES: del FRAME_QUEUES[area_name]
        if area_name in ACTIVE_THREADS: del ACTIVE_THREADS[area_name]
        return

    frame_count = 0
    
    try:
        while area_name in STREAM_SOURCES: 
            ret, frame = cap.read()
            if not ret:
                print(f"Stream ended for {area_name} (End of source).")
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
                
                try:
                    FRAME_QUEUES[area_name].put_nowait(frame_bytes)
                except queue.Full:
                    print(f"Warning: Dropped frame for {area_name} (Queue full).")

                frame_count += 1
                
    except Exception as e:
        print(f"Continuous analysis error for {area_name}: {e}")
        THREAD_RESULTS[area_name] = f"Error: {e}"
        
    finally:
        cap.release()
        
        if area_name in STREAM_SOURCES:
            del STREAM_SOURCES[area_name] 
        if area_name in LIVE_COUNTS:
            del LIVE_COUNTS[area_name]
        if area_name in FRAME_QUEUES:
            del FRAME_QUEUES[area_name]
        if area_name in ACTIVE_THREADS:
            del ACTIVE_THREADS[area_name]
            
        THREAD_RESULTS[area_name] = f"Analysis thread ended. Frames processed: {frame_count}."
        print(f"Finished processing and stored result for {area_name}")

def video_feed_generator(area_name):
    global FRAME_QUEUES
    
    if area_name not in FRAME_QUEUES:
        return

    q = FRAME_QUEUES[area_name]
    
    while area_name in STREAM_SOURCES: 
        try:
            frame_bytes = q.get(timeout=0.1) 
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            q.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Video feed error for {area_name}: {e}")
            break

def start_stream_thread(area_name, source_path):
    global STREAM_SOURCES, FRAME_QUEUES, ACTIVE_THREADS
    
    STREAM_SOURCES[area_name] = source_path
    LIVE_COUNTS[area_name] = 0
    FRAME_QUEUES[area_name] = queue.Queue(maxsize=5) 
    
    thread = threading.Thread(
        target=continuous_video_analysis, 
        args=(source_path, area_name),
        daemon=True 
    )
    thread.start()
    ACTIVE_THREADS[area_name] = thread


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
    camera_data = [
        {'areaName': name, 'sourceType': 'Webcam' if STREAM_SOURCES[name] == '0' else 'Video File'}
        for name in STREAM_SOURCES.keys()
    ]
    
    return render_template('home.html', cameras=camera_data)

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
        return "No active stream for this area. It might have stopped automatically.", 404

    return Response(
        video_feed_generator(area_name),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/get_total_count', methods=['GET'])
def get_total_count():
    total_count = 0
    for count in LIVE_COUNTS.values():
        if isinstance(count, int):
            total_count += count
            
    return jsonify({
        "totalCount": total_count,
        "activeStreams": len(STREAM_SOURCES)
    })

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

    start_stream_thread(area_name, full_path)
    
    return jsonify({"message": f"Video source saved and processing started for {area_name}."}), 200

@app.route('/upload_cam', methods=['POST'])
def upload_cam():
    area_name = request.form.get("areaName", "Unknown")
    
    if area_name in STREAM_SOURCES:
        return jsonify({"message": f"A stream is already active for {area_name}. Stop it first."}), 409

    webcam_source = "0" 
    
    start_stream_thread(area_name, webcam_source)

    return jsonify({"message": f"Camera source saved and processing started for {area_name}."}), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global STREAM_SOURCES, LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS
    area_name = request.json.get("areaName")
    
    if area_name in STREAM_SOURCES:
        del STREAM_SOURCES[area_name]
        
        if area_name in LIVE_COUNTS: del LIVE_COUNTS[area_name]
        if area_name in FRAME_QUEUES: del FRAME_QUEUES[area_name]
        
        message = f"Stream stop signal sent for {area_name}. Analysis thread will terminate shortly."
    else:
        message = "No active stream found to stop."
    
    return jsonify({"message": message}), 200


@app.route('/signout')
def signout():
    global STREAM_SOURCES, LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS
    
    active_streams = list(STREAM_SOURCES.keys())
    for area_name in active_streams:
        if area_name in STREAM_SOURCES: del STREAM_SOURCES[area_name]
        if area_name in LIVE_COUNTS: del LIVE_COUNTS[area_name]
        if area_name in FRAME_QUEUES: del FRAME_QUEUES[area_name]
        
    cv2.destroyAllWindows()
    
    session.pop('user', None)
    response = make_response(redirect(url_for('signin')))
    response.set_cookie('server_jwt', '', expires=0)
    return response

if __name__ == '__main__':
    app.run(debug=True, threaded=True)