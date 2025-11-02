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
from datetime import datetime, timedelta, UTC
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.dates as mdates

CROWD_LOG = [] 
MAX_LOG_SIZE = 500 

FRAME_QUEUES = {} 
ACTIVE_THREADS = {} 
STREAM_SOURCES = {} 
THREAD_RESULTS = {}
LIVE_COUNTS = {} 
OBJECT_TRACKER = {} 
NEXT_ID = 1

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

def get_bbox_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_in_aoi_below_line(center_x, center_y, line_coords):
    x1, y1, x2, y2 = line_coords

    if x1 == x2:
        
        
        return False
        
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    y_on_line = m * center_x + b
    
    
    return center_y > y_on_line

def continuous_video_analysis(source, area_name):
    global LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS, STREAM_SOURCES, OBJECT_TRACKER, NEXT_ID
    
    cap = cv2.VideoCapture(int(source) if source == "0" else source)

    if not cap.isOpened():
        print(f"Error: Cannot open source for {area_name}")
        if area_name in STREAM_SOURCES: del STREAM_SOURCES[area_name]
        if area_name in LIVE_COUNTS: del LIVE_COUNTS[area_name]
        if area_name in FRAME_QUEUES: del FRAME_QUEUES[area_name]
        if area_name in ACTIVE_THREADS: del ACTIVE_THREADS[area_name]
        if area_name in OBJECT_TRACKER: del OBJECT_TRACKER[area_name]
        return

    frame_count = 0
    
    if area_name not in OBJECT_TRACKER:
        OBJECT_TRACKER[area_name] = {}
        
    tracker = OBJECT_TRACKER[area_name]
    
    live_count_in_aoi = 0
    
    while area_name in STREAM_SOURCES: 
        ret, frame = cap.read()
        if not ret:
            print(f"Stream ended for {area_name} (End of source).")
            break 
            
        source_info = STREAM_SOURCES.get(area_name, {})
        line_coords = source_info.get('line_coords', None)
        
        results = model.predict(
            source=frame,
            classes=[person_class_id], 
            conf=confidence_threshold,
            iou=iou_threshold,
            stream=False, 
            verbose=False,
        )

        person_count = 0
        live_count_in_aoi = 0
        
        try:
            result = next(iter(results))
        except StopIteration:
            result = None
            
        if result is not None and result.orig_img is not None:
            processed_frame = result.orig_img
            person_count = len(result.boxes)
            
            new_tracker = {}
            active_ids = set()
            
            for i, box in enumerate(result.boxes):
                x_1, y_1, x_2, y_2 = map(int, box.xyxy[0].tolist())
                center_x, center_y = get_bbox_center(x_1, y_1, x_2, y_2)
                
                
                match_id = -1
                min_dist = float('inf')
                
                for obj_id, obj_data in list(tracker.items()):
                    prev_x, prev_y, is_inside = obj_data 
                    dist = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                    if dist < min_dist and dist < 100: 
                        min_dist = dist
                        match_id = obj_id
                
                
                if match_id == -1:
                    match_id = NEXT_ID
                    NEXT_ID += 1
                    is_inside = False 
                else:
                    prev_x, prev_y, is_inside = tracker[match_id] 
                
                
                
                
                if line_coords and is_in_aoi_below_line(center_x, center_y, line_coords):
                    live_count_in_aoi += 1
                    is_inside = True
                    color = (0, 255, 0) 
                else:
                    is_inside = False
                    color = (255, 0, 255) 
                
                
                new_tracker[match_id] = (center_x, center_y, is_inside)
                active_ids.add(match_id)
                
                
                label = str(match_id)
                cv2.rectangle(processed_frame, (x_1, y_1), (x_2, y_2), color, 2)
                cv2.putText(processed_frame, label, (x_1, y_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            
            tracker.clear()
            tracker.update(new_tracker)
            
            
            keys_to_delete = [obj_id for obj_id in tracker.keys() if obj_id not in active_ids]
            for obj_id in keys_to_delete:
                del tracker[obj_id]
                
            
            
            if line_coords:
                x1, y1, x2, y2 = line_coords
                line_color = (0, 0, 255) 
                line_thickness = 3
                cv2.line(processed_frame, (x1, y1), (x2, y2), line_color, line_thickness)
                
                LIVE_COUNTS[area_name] = live_count_in_aoi
                
                cv2.putText(processed_frame, f'TOTAL IN AOI: {live_count_in_aoi}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            else:
                
                LIVE_COUNTS[area_name] = person_count
                
            
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            
            if area_name in FRAME_QUEUES:
                try:
                    FRAME_QUEUES[area_name].put_nowait(frame_bytes)
                except queue.Full:
                    print(f"Warning: Dropped frame for {area_name} (Queue full).")
            
            time.sleep(0.01)

            frame_count += 1
                
        else:
            
            time.sleep(0.01)

    cap.release()
    
    if area_name in STREAM_SOURCES:
        del STREAM_SOURCES[area_name] 
    if area_name in LIVE_COUNTS:
        del LIVE_COUNTS[area_name]
    if area_name in FRAME_QUEUES:
        del FRAME_QUEUES[area_name]
    if area_name in ACTIVE_THREADS:
        del ACTIVE_THREADS[area_name]
    if area_name in OBJECT_TRACKER:
        del OBJECT_TRACKER[area_name]
        
    THREAD_RESULTS[area_name] = f"Analysis thread ended. Frames processed: {frame_count}."
    print(f"Finished processing and stored result for {area_name}")


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

def video_feed_generator(area_name):
    global FRAME_QUEUES
    
    if area_name not in FRAME_QUEUES:
        return

    q = FRAME_QUEUES[area_name]
    
    while area_name in STREAM_SOURCES: 
        try:
            frame_bytes = q.get(timeout=0.1) 
            
            yield (b'--frame_boundary\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            q.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Video feed error for {area_name}: {e}")
            break

def start_stream_thread(area_name, source_path, threshold, line_coords=None):
    global STREAM_SOURCES, FRAME_QUEUES, ACTIVE_THREADS, OBJECT_TRACKER
    
    STREAM_SOURCES[area_name] = {'source': source_path, 'threshold': threshold, 'line_coords': line_coords}
    LIVE_COUNTS[area_name] = 0
    FRAME_QUEUES[area_name] = queue.Queue(maxsize=15) 
    OBJECT_TRACKER[area_name] = {}
    
    thread = threading.Thread(
        target=continuous_video_analysis, 
        args=(source_path, area_name),
        daemon=True 
    )
    thread.start()
    ACTIVE_THREADS[area_name] = thread

def generate_crowd_chart_image():
    global CROWD_LOG
    
    data_by_area = {}
    
    limited_log = CROWD_LOG[-15:]
    
    for entry in limited_log:
        try:
            entry_time = datetime.fromisoformat(entry['time'].replace('Z', '+00:00'))
            area = entry['area']
            if area not in data_by_area:
                data_by_area[area] = {'times': [], 'counts': []}
            
            data_by_area[area]['times'].append(entry_time)
            data_by_area[area]['counts'].append(entry['count'])
        except ValueError:
            continue
            
    if not data_by_area:
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))
    
    for area, data in data_by_area.items():
        ax.plot(data['times'], data['counts'], label=area, marker='o', markersize=4)

    ax.set_xlabel('Time')
    ax.set_ylabel('Number of People')
    ax.grid(True)
    ax.legend(loc='upper left')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig) 
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return data

def generate_density_chart_image():
    global LIVE_COUNTS, STREAM_SOURCES
    
    areas = []
    densities = []
    colors = []
    
    for area_name, current_count in LIVE_COUNTS.items():
        stream_info = STREAM_SOURCES.get(area_name)
        
        if stream_info and isinstance(current_count, int):
            threshold = stream_info.get('threshold')
            if threshold and threshold > 0:
                density_percentage = (current_count / threshold) * 100
                
                if density_percentage < 50:
                    color = 'green'
                elif density_percentage < 90:
                    color = 'gold'
                else:
                    color = 'red'
                
                areas.append(area_name)
                densities.append(density_percentage)
                colors.append(color)

    if not areas:
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))
    
    bars = ax.bar(areas, densities, color=colors)

    ax.set_ylabel('Density Percentage (%)')
    ax.set_ylim(0, 110) 
    ax.grid(axis='y', linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., 
                height + 3,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig) 
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    
    return data


app = Flask(__name__)
app.secret_key = 'a-very-secret-key'
JWT_SECRET = 'another-very-secret-key'
JWT_ALGORITHM = 'HS256'

def get_user_data(uid, id_token):
    return db.child("user").child(uid).get(id_token).val()

def generate_server_jwt(uid, email, username):
    current_time = datetime.now(UTC) 
    payload = {
        "uid": uid,
        "email": email,
        "username": username,
        "iat": current_time,
        "exp": current_time + timedelta(hours=1)
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
        {'areaName': name, 
         'sourceType': 'Webcam' if STREAM_SOURCES[name]['source'] == '0' else 'Video File',
         'capacityThreshold': STREAM_SOURCES[name].get('threshold', 'N/A')}
        for name in STREAM_SOURCES.keys()
    ]
    
    historical_chart_data = generate_crowd_chart_image() 
    density_chart_data = generate_density_chart_image() 
    
    return render_template('home.html', 
                           cameras=camera_data,
                           historical_chart_data=historical_chart_data,
                           density_chart_data=density_chart_data)

@app.route('/vid_analy')
def vid_analy():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
        
    camera_data = [
        {'areaName': name, 'sourceType': 'Webcam' if STREAM_SOURCES[name]['source'] == '0' else 'Video File'}
        for name in STREAM_SOURCES.keys()
    ]

    return render_template("vid_analy.html", cameras=camera_data)


@app.route('/cam_manage', methods=['GET'])
def cam_manage_page():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    return render_template("cam_manage.html")

@app.route('/define_aoi/<area_name>')
def define_aoi_page(area_name):
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    
    if area_name not in STREAM_SOURCES:
        return "Error: Stream must be active to define AOI.", 400
        
    return render_template('define_aoi.html', area_name=area_name)

@app.route('/video_feed/<area_name>')
def video_feed(area_name):
    source_info = STREAM_SOURCES.get(area_name)
    if not source_info:
        return "No active stream for this area. It might have stopped automatically.", 404

    return Response(
        video_feed_generator(area_name),
        mimetype='multipart/x-mixed-replace; boundary=frame_boundary'
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
    
    threshold = STREAM_SOURCES.get(area_name, {}).get('threshold', 'N/A')

    return jsonify({
        "areaName": area_name,
        "count": current_count,
        "isActive": is_active,
        "threshold": threshold
    })


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video_file = request.files["video"]
    area_name = request.form.get("areaName", "Unknown")
    capacity_threshold = int(request.form.get("capacityThreshold", 0))
    
    if area_name in STREAM_SOURCES:
        return jsonify({"message": f"A stream is already active for {area_name}. Stop it first."}), 409
        
    filename = secure_filename(video_file.filename)
    full_path = os.path.join(UPLOAD_FOLDER, filename)
    video_file.save(full_path)
    
    if not os.path.exists(full_path):
        return jsonify({"error": f"Failed to save video at: {full_path}"}), 500

    start_stream_thread(area_name, full_path, capacity_threshold)
    
    return jsonify({"message": f"Video source saved and processing started for {area_name}."}), 200

@app.route('/upload_cam', methods=['POST'])
def upload_cam():
    area_name = request.form.get("areaName", "Unknown")
    capacity_threshold = int(request.form.get("capacityThreshold", 0))
    
    if area_name in STREAM_SOURCES:
        return jsonify({"message": f"A stream is already active for {area_name}. Stop it first."}), 409

    webcam_source = "0" 
    
    start_stream_thread(area_name, webcam_source, capacity_threshold)

    return jsonify({"message": f"Camera source saved and processing started for {area_name}."}), 200

@app.route('/set_aoi', methods=['POST'])
def set_aoi():
    data = request.get_json()
    area_name = data.get("areaName")
    line_coords = data.get("line_coords") 

    if not area_name or not line_coords or len(line_coords) != 4:
        return jsonify({"message": "Invalid data for setting AOI."}), 400

    if area_name not in STREAM_SOURCES:
        return jsonify({"message": f"Stream for {area_name} not found or inactive. Start stream first."}), 404

    stream_info = STREAM_SOURCES[area_name]
    stream_info['line_coords'] = line_coords

    return jsonify({"message": f"AOI line coordinates set successfully for {area_name}."}), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global STREAM_SOURCES, LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS, OBJECT_TRACKER
    area_name = request.json.get("areaName")
    
    if area_name in STREAM_SOURCES:
        del STREAM_SOURCES[area_name]
        
        if area_name in LIVE_COUNTS: del LIVE_COUNTS[area_name]
        if area_name in FRAME_QUEUES: del FRAME_QUEUES[area_name]
        if area_name in OBJECT_TRACKER: del OBJECT_TRACKER[area_name]
        
        message = f"Stream stop signal sent for {area_name}. Analysis thread will terminate shortly."
    else:
        message = "No active stream found to stop."
    
    return jsonify({"message": message}), 200

@app.route('/signout')
def signout():
    global STREAM_SOURCES, LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS, OBJECT_TRACKER
    
    active_streams = list(STREAM_SOURCES.keys())
    for area_name in active_streams:
        if area_name in STREAM_SOURCES: del STREAM_SOURCES[area_name]
        if area_name in LIVE_COUNTS: del LIVE_COUNTS[area_name]
        if area_name in FRAME_QUEUES: del FRAME_QUEUES[area_name]
        if area_name in OBJECT_TRACKER: del OBJECT_TRACKER[area_name]
        
    cv2.destroyAllWindows()
    
    session.pop('user', None)
    response = make_response(redirect(url_for('signin')))
    response.set_cookie('server_jwt', '', expires=0)
    return response

@app.route('/log_area_count', methods=['POST'])
def log_area_count():
    global CROWD_LOG
    data = request.get_json()
    
    if not data or 'area' not in data or 'count' not in data or 'time' not in data:
        return jsonify({"message": "Invalid data payload. Missing area, count, or time."}), 400
        
    area_name = data['area']
    count = data['count']
    timestamp = data['time'] 
    
    log_entry = {
        'area': area_name,
        'count': int(count),
        'time': timestamp
    }
    
    CROWD_LOG.append(log_entry)
    
    if len(CROWD_LOG) > MAX_LOG_SIZE:
        CROWD_LOG = CROWD_LOG[-MAX_LOG_SIZE:]
    
    return jsonify({
        "message": "Count and Time logged successfully", 
        "area": area_name
    }), 200

@app.route('/get_historical_log', methods=['GET'])
def get_historical_log():
    global CROWD_LOG
    five_minutes_ago = datetime.now(UTC) - timedelta(minutes=5)
    
    recent_log = []
    for entry in CROWD_LOG:
        try:
            entry_time = datetime.fromisoformat(entry['time'].replace('Z', '+00:00'))
            if entry_time > five_minutes_ago:
                recent_log.append(entry)
        except ValueError:
            continue

    return jsonify(recent_log)

@app.route('/get_server_chart_data', methods=['GET'])
def get_server_chart_data():
    historical_data = generate_crowd_chart_image() 
    density_data = generate_density_chart_image()
    
    if not historical_data and not density_data:
        return jsonify({"message": "No chart data available."}), 204
        
    return jsonify({
        "historical_chart_data": historical_data,
        "density_chart_data": density_data
    })


if __name__ == '__main__':
    app.run(debug=True, threaded=True)