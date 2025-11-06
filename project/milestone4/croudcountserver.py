from flask import Flask, render_template, redirect, url_for, request, session, make_response, jsonify, Response
import datetime
import pyrebase
import jwt
import os
import csv
import io
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
import base64
import matplotlib.dates as mdates

app = Flask(__name__)
app.secret_key = 'a-very-secret-key'
JWT_SECRET = 'another-very-secret-key'
JWT_ALGORITHM = 'HS256'

# --- Configuration & Initialization ---

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

# YOLO Model Setup
model = YOLO("yolov8n.pt")
person_class_id = 0
confidence_threshold = 0.4
iou_threshold = 0.7

# --- Global State (Per-user/per-area state using composite key) ---

USER_LOGS_REF = "user_logs"
USER_CAMERAS_REF = "user_cameras"
GUEST_UID = "GUEST_USER_SESSION"
MAX_LOG_SIZE = 500
NEXT_ID = 1

# DUMMY TOKEN FOR GUEST AUTHENTICATION BYPASS (Must be non-empty)
GUEST_ID_TOKEN = "DUMMY_TOKEN_FOR_AUTHENTICATED_READ"

STREAM_SOURCES = {}
FRAME_QUEUES = {}
ACTIVE_THREADS = {}
THREAD_RESULTS = {}
LIVE_COUNTS = {}
OBJECT_TRACKER = {}

# =======================================================
# --- AUTHENTICATION AND DATA UTILITY FUNCTIONS (Defined FIRST) ---
# =======================================================

def generate_server_jwt(uid, email, username):
    """Generates a secure server-side JWT for cookie-based session management."""
    payload = {
        "uid": uid, "email": email, "username": username,
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + timedelta(hours=1)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_server_jwt(token):
    """Verifies and decodes the server JWT."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def check_auth(request):
    """Checks for a valid server JWT in the cookie."""
    return verify_server_jwt(request.cookies.get('server_jwt'))

def get_logged_in_uid(request):
    """Retrieves the UID from the server JWT cookie."""
    jwt_token = request.cookies.get('server_jwt')
    decoded = verify_server_jwt(jwt_token)
    return decoded.get('uid') if decoded else None

def get_auth_token(request):
    """Retrieves the Firebase ID token from the session."""
    return session['user'].get('idToken') if 'user' in session else None

def get_display_uid_and_token(request):
    """Determines the effective UID and token for displaying data (handles admin view)."""
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid:
        return None, None, False
    logged_in_token = get_auth_token(request)
    
    is_admin_view = (logged_in_uid != GUEST_UID) and ('VIEWING_UID' in session) and (session.get('VIEWING_UID') != logged_in_uid)
    display_uid = session.get('VIEWING_UID') if is_admin_view else logged_in_uid
    
    return display_uid, logged_in_token, is_admin_view

def get_effective_fetch_token(token, display_uid):
    """Determines the token to use for DB fetching based on Guest/Admin status."""
    if token == GUEST_ID_TOKEN:
        # If the viewer is the Guest, and they are trying to read any Firebase data,
        # we must return a genuinely empty string to signal an unauthenticated read to pyrebase.
        return ""
    # Otherwise, return the real token for authenticated reads
    return token

def fetch_data_from_db(path, token=""):
    """
    Robustly fetches data using the positional token argument for older pyrebase versions.
    """
    try:
        if token and token.strip():
            return db.child(path).get(token).val()
        else:
            return db.child(path).get().val()
    except Exception as e:
        print(f"CRITICAL FETCH ERROR accessing DB path {path}: {e}")
        return None

def get_user_data(uid, id_token):
    """Fetches user data using positional token argument (compatible with pyrebase)."""
    path = f"user/{uid}"
    return fetch_data_from_db(path, id_token)

def get_all_users(id_token, current_user_uid):
    """Fetches list of all users for display."""
    users_data = fetch_data_from_db("user", id_token)
    
    if users_data is None:
        return []
            
    return [{
        'uid': uid,
        'username': data.get('user_name', 'Unknown User')
    } for uid, data in users_data.items() if uid != GUEST_UID and uid != current_user_uid and 'user_name' in data]

def get_logged_areas(display_uid, token):
    """
    Fetches areas with logged data for the displayed user. 
    Handles GUEST Admin View data fetching via DUMMY TOKEN.
    """
    if display_uid == GUEST_UID:
        return sorted(set(entry['area'] for entry in session.get('GUEST_LOGS', [])))
    
    effective_token = get_effective_fetch_token(token, display_uid)

    path = f"{USER_LOGS_REF}/{display_uid}"
    all_logs = fetch_data_from_db(path, effective_token) or {}
    
    logged_areas = [key for key, value in all_logs.items() if isinstance(value, dict) and value]
    
    return sorted(logged_areas)

# =======================================================
# --- VIDEO PROCESSING & STREAM MANAGEMENT FUNCTIONS ---
# =======================================================

def get_bbox_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def is_in_aoi_below_line(center_x, center_y, line_coords):
    x1, y1, x2, y2 = line_coords
    
    if x1 == x2: return False
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return center_y > m * center_x + b

def cleanup(uid, area_name):
    key = (uid, area_name)
    for d in [LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS, OBJECT_TRACKER, THREAD_RESULTS]: d.pop(key, None)
    if uid in STREAM_SOURCES and area_name in STREAM_SOURCES[uid]: STREAM_SOURCES[uid].pop(area_name)
    if uid in STREAM_SOURCES and not STREAM_SOURCES[uid]: STREAM_SOURCES.pop(uid)

def get_stream_info(uid, area_name):
    return STREAM_SOURCES.get(uid, {}).get(area_name)

def get_all_user_streams(uid):
    return STREAM_SOURCES.get(uid, {})

def start_stream_thread(uid, area_name, source_path, threshold, line_coords=None):
    global STREAM_SOURCES, FRAME_QUEUES, OBJECT_TRACKER
    STREAM_SOURCES.setdefault(uid, {})[area_name] = {'source': source_path, 'threshold': threshold, 'line_coords': line_coords, 'start_time': datetime.now().isoformat()}
    key = (uid, area_name)
    LIVE_COUNTS[key] = 0
    FRAME_QUEUES[key] = queue.Queue(maxsize=15)
    OBJECT_TRACKER[key] = {}
    thread = threading.Thread(target=continuous_video_analysis, args=(source_path, area_name, uid), daemon=True)
    thread.start()
    ACTIVE_THREADS[key] = thread

def continuous_video_analysis(source, area_name, owner_uid):
    global LIVE_COUNTS, FRAME_QUEUES, ACTIVE_THREADS, OBJECT_TRACKER, NEXT_ID
    key = (owner_uid, area_name)
    cap = cv2.VideoCapture(int(source) if source == "0" else source)
    if not cap.isOpened():
        print(f"Error: Cannot open source for {area_name} (User: {owner_uid})")
        cleanup(owner_uid, area_name)
        return
    frame_count = 0
    tracker = OBJECT_TRACKER.get(key, {}) 
    if not tracker: OBJECT_TRACKER[key] = {}; tracker = OBJECT_TRACKER[key]
    while get_stream_info(owner_uid, area_name):
        ret, frame = cap.read()
        if not ret: print(f"Stream ended for {area_name} (User: {owner_uid})"); break
        source_info = get_stream_info(owner_uid, area_name)
        if not source_info: break
        line_coords = source_info.get('line_coords')
        results = model.predict(source=frame, classes=[person_class_id], conf=confidence_threshold, iou=iou_threshold, stream=False, verbose=False)
        result = next(iter(results), None)
        if not result or result.orig_img is None: time.sleep(0.01); continue
        processed_frame = result.orig_img.copy()
        person_count = 0; live_count_in_aoi = 0; new_tracker = {}; active_ids = set()
        if result.boxes:
            for i, box in enumerate(result.boxes):
                x_1, y_1, x_2, y_2 = map(int, box.xyxy[0].tolist())
                center_x, center_y = get_bbox_center(x_1, y_1, x_2, y_2)
                match_id = -1; min_dist = float('inf')
                for obj_id, obj_data in list(tracker.items()):
                    prev_x, prev_y, is_inside = obj_data 
                    dist = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
                    if dist < min_dist and dist < 100: min_dist = dist; match_id = obj_id
                if match_id == -1:
                    global NEXT_ID 
                    match_id = NEXT_ID; NEXT_ID += 1; is_inside = False 
                else:
                    if match_id in tracker: prev_x, prev_y, is_inside = tracker[match_id] 
                    else: is_inside = False
                person_count += 1
                if line_coords and is_in_aoi_below_line(center_x, center_y, line_coords):
                    live_count_in_aoi += 1; is_inside = True; color = (0, 255, 0)
                else: is_inside = False; color = (255, 0, 255)
                new_tracker[match_id] = (center_x, center_y, is_inside); active_ids.add(match_id)
                label = str(match_id)
                cv2.rectangle(processed_frame, (x_1, y_1), (x_2, y_2), color, 2)
                cv2.putText(processed_frame, label, (x_1, y_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            tracker.clear(); tracker.update(new_tracker)
            keys_to_delete = [obj_id for obj_id in tracker.keys() if obj_id not in active_ids]
            for obj_id in keys_to_delete: del tracker[obj_id]
        if line_coords:
            x1, y1, x2, y2 = line_coords
            cv2.line(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            LIVE_COUNTS[key] = live_count_in_aoi
            cv2.putText(processed_frame, f'TOTAL IN AOI: {live_count_in_aoi}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            LIVE_COUNTS[key] = person_count
            cv2.putText(processed_frame, f'TOTAL DETECTED: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        if key in FRAME_QUEUES:
            try: FRAME_QUEUES[key].put_nowait(frame_bytes)
            except queue.Full: pass 
        time.sleep(0.01)
        frame_count += 1
    cap.release()
    cleanup(owner_uid, area_name)
    THREAD_RESULTS[key] = f"Analysis ended. Frames processed: {frame_count}"

def video_feed_generator(uid, area_name):
    key = (uid, area_name)
    if key not in FRAME_QUEUES: return
    q = FRAME_QUEUES[key]
    while get_stream_info(uid, area_name):
        try:
            frame_bytes = q.get(timeout=0.1)
            yield (b'--frame_boundary\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            q.task_done()
        except queue.Empty: continue
        except Exception as e: print(f"Video feed error for {area_name} (User: {uid}): {e}"); break

# =======================================================
# --- CHART GENERATION FUNCTIONS ---
# =======================================================

def generate_crowd_chart_image():
    """Generates a historical crowd count chart based on logged data."""
    display_uid, token, _ = get_display_uid_and_token(request)
    if not display_uid:
        return ""

    log_entries = []
    
    effective_token = get_effective_fetch_token(token, display_uid)

    if display_uid == GUEST_UID:
        log_entries = session.get('GUEST_LOGS', [])
    else:
        # Pass effective_token for charts
        all_logs = fetch_data_from_db(f"{USER_LOGS_REF}/{display_uid}", effective_token) or {}
        log_entries = [entry for area in all_logs.values() if isinstance(area, dict) for entry in area.values()]

    if not log_entries:
        return ""

    log_entries.sort(key=lambda x: x.get('time', ''))
    limited_log = log_entries[-15:]
    data_by_area = {}

    for entry in limited_log:
        try:
            t = datetime.fromisoformat(entry['time'].replace('Z', '+00:00'))
            area = entry['area']
            data_by_area.setdefault(area, {'times': [], 'counts': []})
            data_by_area[area]['times'].append(t)
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
    ax.set_title('Historical Crowd Count (Last 15 Entries)')
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getbuffer()).decode("ascii")

def generate_density_chart_image():
    """Generates a live density bar chart for active streams (user-specific)."""
    display_uid, token, _ = get_display_uid_and_token(request)
    if not display_uid: return ""

    user_streams = get_all_user_streams(display_uid)

    areas, densities, colors = [], [], []
    for area_name, info in user_streams.items():
        key = (display_uid, area_name)
        count = LIVE_COUNTS.get(key)
        
        if info and isinstance(count, int) and (threshold := info.get('threshold')) and threshold > 0:
            density = min((count / threshold) * 100, 100)
            color = 'green' if density < 50 else 'gold' if density < 90 else 'red'
            areas.append(area_name)
            densities.append(density)
            colors.append(color)

    if not areas:
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(areas, densities, color=colors)
    ax.set_ylabel('Density Percentage (%)')
    ax.set_title(f'Live Density for {display_uid}' if display_uid != GUEST_UID else 'Live Density for Guest')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', linestyle='--')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 3, f'{h:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getbuffer()).decode("ascii")

# =======================================================
# --- FLASK ROUTES ---
# =======================================================

@app.route('/')
def root():
    return redirect(url_for('signin'))

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/login', methods=['POST'])
def login():
    if request.form.get('is_guest') == 'true':
        server_jwt = generate_server_jwt(GUEST_UID, 'guest@session.com', 'Guest')
        session['user'] = {'uid': GUEST_UID, 'idToken': GUEST_ID_TOKEN, 'jwt': server_jwt}
        session.pop('VIEWING_UID', None)
        session['GUEST_LOGS'] = []
        response = make_response(redirect(url_for('index')))
        response.set_cookie('server_jwt', server_jwt, httponly=True, samesite='Lax')
        return response

    try:
        email, password = request.form['email'], request.form['password']
        user = auth.sign_in_with_email_and_password(email, password)
        uid, id_token = user['localId'], user['idToken']
        
        user_data = get_user_data(uid, id_token)
        if user_data is None:
            print(f"Warning: User {uid} signed in but profile data is missing. Logging out.")
            return render_template('signin.html', Error="Profile data error. Please register again.")
            
        username = user_data.get('user_name', 'User')
        server_jwt = generate_server_jwt(uid, email, username)
        session['user'] = {'uid': uid, 'idToken': id_token, 'jwt': server_jwt}
        session.pop('VIEWING_UID', None)
        response = make_response(redirect(url_for('index')))
        response.set_cookie('server_jwt', server_jwt, httponly=True, samesite='Lax')
        return response
    except Exception as e:
        print(e)
        return render_template('signin.html', Error="Invalid username/password!!")

@app.route('/register', methods=['POST'])
def register():
    try:
        email, password, username = request.form['email'], request.form['password'], request.form['username']
        user = auth.create_user_with_email_and_password(email, password)
        uid, id_token = user['localId'], user['idToken']
        db.child("user").child(uid).set({"user_name": username}, id_token)
        return redirect(url_for('signin'))
    except Exception as e:
        print(e)
        return render_template('signin.html', Error="Could not register. Email may already be in use.", mode='register')

@app.route('/index')
def index():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    _, _, is_admin_view = get_display_uid_and_token(request)
    return render_template('index.html', username=decoded.get('username', 'User'), is_admin_view=is_admin_view)

@app.route('/view_user/<target_uid>')
def view_user(target_uid):
    """
    Allows Guest users (or any logged-in user) to access the Admin View.
    """
    logged_in_uid = get_logged_in_uid(request)
    
    if not logged_in_uid:
        return redirect(url_for('signin'))
    
    if logged_in_uid != target_uid:
        session['VIEWING_UID'] = target_uid
    else:
        session.pop('VIEWING_UID', None) 
        
    return redirect(url_for('home'))

@app.route('/exit_user_view')
def exit_user_view():
    session.pop('VIEWING_UID', None)
    return redirect(url_for('home'))

@app.route('/profile')
def profile():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    if decoded.get('uid') == GUEST_UID:
        return render_template('profile.html', username='Guest', email='Session Only')
    return render_template('profile.html', username=decoded.get('username'), email=decoded.get('email'))

@app.route('/home')
def home():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))

    logged_in_uid = decoded.get('uid')
    token = get_auth_token(request)
    display_uid, _, is_admin_view = get_display_uid_and_token(request)

    # --- Determine the token to use for data fetching ---
    effective_token = token
    if logged_in_uid == GUEST_UID:
        effective_token = GUEST_ID_TOKEN
    # --------------------------------------------------

    displayed_username = (get_user_data(display_uid, effective_token).get('user_name', 'Viewed User')
                          if is_admin_view and display_uid else decoded.get('username', 'User'))

    # FIX: Only display user list if the user is a registered user, not Guest.
    if logged_in_uid != GUEST_UID and not is_admin_view:
         all_users = get_all_users(token, logged_in_uid)
    else:
         all_users = []
    
    # Fetch streams and charts using the determined token
    current_user_streams = STREAM_SOURCES.get(display_uid, {}) # Stream list is global state, no token needed here
    
    historical_chart_data = generate_crowd_chart_image()
    density_chart_data = generate_density_chart_image()


    camera_data = [
        {'areaName': name,
         'sourceType': 'Webcam' if info['source'] == '0' else 'Video File',
         'capacityThreshold': info.get('threshold', 'N/A')}
        for name, info in current_user_streams.items()
    ]

    return render_template('home.html',
                           cameras=camera_data,
                           historical_chart_data=historical_chart_data,
                           density_chart_data=density_chart_data,
                           all_users=all_users,
                           is_admin_view=is_admin_view,
                           displayed_username=displayed_username)

@app.route('/history_download')
def history_download_page():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    display_uid, token, is_admin_view = get_display_uid_and_token(request)
    
    # --- Manual Token Substitution for Admin View ---
    effective_token = token
    if get_logged_in_uid(request) == GUEST_UID:
        effective_token = GUEST_ID_TOKEN
    # -----------------------------------------------

    logged_areas = get_logged_areas(display_uid, effective_token) 
    
    displayed_username = (get_user_data(display_uid, effective_token).get('user_name', 'Viewed User')
                          if is_admin_view and display_uid != GUEST_UID else decoded.get('username', 'User'))
                          
    return render_template('history_download.html', logged_areas=logged_areas,
                           is_admin_view=is_admin_view, displayed_username=displayed_username)

@app.route('/vid_analy')
def vid_analy():
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    logged_in_uid = decoded.get('uid')

    if get_display_uid_and_token(request)[2]:
        return redirect(url_for('home'))

    current_user_streams = get_all_user_streams(logged_in_uid)
    camera_data = [{'areaName': name,
                    'sourceType': 'Webcam' if info['source'] == '0' else 'Video File'}
                   for name, info in current_user_streams.items()]
    return render_template("vid_analy.html", cameras=camera_data)

@app.route('/cam_manage', methods=['GET'])
def cam_manage_page():
    if check_auth(request) is None:
        return redirect(url_for('signin'))
    if get_display_uid_and_token(request)[2]:
        return redirect(url_for('home'))
    return render_template("cam_manage.html")

@app.route('/define_aoi/<area_name>')
def define_aoi_page(area_name):
    decoded = check_auth(request)
    if not decoded:
        return redirect(url_for('signin'))
    logged_in_uid = decoded.get('uid')
    if get_display_uid_and_token(request)[2]:
        return redirect(url_for('home'))

    if not get_stream_info(logged_in_uid, area_name):
        return "Error: Stream must be active to define AOI.", 400
    return render_template('define_aoi.html', area_name=area_name)

@app.route('/video_feed/<area_name>')
def video_feed(area_name):
    display_uid, _, _ = get_display_uid_and_token(request)
    if not display_uid: return "Unauthorized.", 401

    if not get_stream_info(display_uid, area_name):
        return "No active stream.", 404

    return Response(video_feed_generator(display_uid, area_name),
                    mimetype='multipart/x-mixed-replace; boundary=frame_boundary')

@app.route('/download_history/')
@app.route('/download_history/<area_name>')
def download_history(area_name=''):
    display_uid, token, _ = get_display_uid_and_token(request)
    if not display_uid:
        return redirect(url_for('signin'))

    history = []
    # --- Manual Token Substitution for Admin View ---
    effective_token = token
    if get_logged_in_uid(request) == GUEST_UID and display_uid != GUEST_UID:
        effective_token = GUEST_ID_TOKEN
    # -----------------------------------------------

    if display_uid == GUEST_UID:
        history = [e for e in session.get('GUEST_LOGS', []) if e['area'] == area_name] if area_name else session.get('GUEST_LOGS', [])
    elif token:
        all_logs = fetch_data_from_db(f"{USER_LOGS_REF}/{display_uid}", effective_token) or {}

        if area_name:
            history = list(all_logs.get(area_name, {}).values())
        else:
            for area in all_logs.values():
                if isinstance(area, dict):
                    history.extend(area.values())
    else:
        return redirect(url_for('signin'))

    if not history:
        return redirect(url_for('history_download_page')) if not area_name else (jsonify({"message": "No data."}), 404)

    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['Area', 'Count', 'Timestamp'])
    for e in history:
        cw.writerow([e.get('area', 'N/A'), e.get('count', 0), e.get('time', 'N/A')])

    output = make_response(si.getvalue())
    
    if not area_name:
        if display_uid == GUEST_UID:
            filename = "GUEST_ALL_history.csv" 
        else:
            filename = f"{display_uid}_ALL_history.csv"
    else:
        filename = f"{area_name}_history.csv"

    output.headers["Content-Disposition"] = f"attachment; filename={filename}"
    output.headers["Content-type"] = "text/csv"
    return output

@app.route('/delete_history/<area_name>', methods=['POST'])
def delete_history(area_name):
    logged_in_uid = get_logged_in_uid(request)
    token = get_auth_token(request)
    if not logged_in_uid or not token:
        return redirect(url_for('signin'))
    
    if 'VIEWING_UID' in session and session['VIEWING_UID'] != logged_in_uid:
        return jsonify({"message": "Cannot delete history in Admin View."}), 403

    if logged_in_uid == GUEST_UID:
        session['GUEST_LOGS'] = [e for e in session.get('GUEST_LOGS', []) if e['area'] != area_name]
        return jsonify({"message": f"Cleared session history for {area_name}."}), 200
    else:
        try:
            db.child(USER_LOGS_REF).child(logged_in_uid).child(area_name).remove(token)
            return jsonify({"message": f"Deleted history for {area_name}."}), 200
        except Exception as e:
            print(f"Delete error: {e}")
            return jsonify({"message": "Failed to delete history."}), 500

@app.before_request
def check_admin_view_permissions():
    """Guardrail to prevent modification actions when an admin is viewing another user's data."""
    if request.method == 'POST' or request.endpoint in ['cam_manage_page', 'vid_analy', 'define_aoi_page']:
        _, _, is_admin_view = get_display_uid_and_token(request)
        if is_admin_view:
            action_endpoints = ['log_area_count', 'upload_video', 'upload_cam', 'set_aoi', 'stop_stream', 'delete_history', 'api_camera_config']
            
            if request.endpoint in action_endpoints or request.path in ['/upload_video', '/upload_cam', '/set_aoi', '/stop_stream', '/delete_history', '/api/camera_config']:
                return (jsonify({"message": "Restricted in Admin View."}), 403) if request.is_json else redirect(url_for('home'))

@app.route('/get_total_count', methods=['GET'])
def get_total_count():
    display_uid, _, _ = get_display_uid_and_token(request)
    if not display_uid: return jsonify({"totalCount": 0, "activeStreams": 0})

    user_streams = get_all_user_streams(display_uid)
    total_count = sum(LIVE_COUNTS.get((display_uid, name), 0) for name in user_streams.keys())

    return jsonify({"totalCount": total_count, "activeStreams": len(user_streams)})

@app.route('/get_count/<area_name>', methods=['GET'])
def get_count(area_name):
    display_uid, _, _ = get_display_uid_and_token(request)
    if not display_uid: return jsonify({"areaName": area_name, "count": "N/A", "isActive": False, "threshold": "N/A"})

    info = get_stream_info(display_uid, area_name)
    key = (display_uid, area_name)

    return jsonify({
        "areaName": area_name,
        "count": LIVE_COUNTS.get(key, "N/A"),
        "isActive": info is not None,
        "threshold": info.get('threshold', 'N/A') if info else 'N/A'
    })

@app.route('/upload_video', methods=['POST'])
def upload_video():
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid: return jsonify({"error": "Unauthorized"}), 401

    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    video_file = request.files["video"]
    area_name = request.form.get("areaName", "Unknown")
    threshold = int(request.form.get("capacityThreshold", 0))

    if get_stream_info(logged_in_uid, area_name):
        return jsonify({"message": "Stream active. Stop first."}), 409

    path = os.path.join(UPLOAD_FOLDER, secure_filename(video_file.filename))
    video_file.save(path)
    if not os.path.exists(path):
        return jsonify({"error": "Save failed"}), 500

    start_stream_thread(logged_in_uid, area_name, path, threshold)
    return jsonify({"message": f"Processing started for {area_name}."}), 200

@app.route('/upload_cam', methods=['POST'])
def upload_cam():
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid: return jsonify({"error": "Unauthorized"}), 401

    area_name = request.form.get("areaName", "Unknown")
    threshold = int(request.form.get("capacityThreshold", 0))

    if get_stream_info(logged_in_uid, area_name):
        return jsonify({"message": "Stream active."}), 409

    start_stream_thread(logged_in_uid, area_name, "0", threshold)
    return jsonify({"message": f"Camera started for {area_name}."}), 200

@app.route('/set_aoi', methods=['POST'])
def set_aoi():
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid: return jsonify({"message": "Unauthorized."}), 401

    data = request.get_json()
    area_name = data.get("areaName")
    line_coords = data.get("line_coords")
    if not area_name or not line_coords or len(line_coords) != 4:
        return jsonify({"message": "Invalid AOI data."}), 400

    if not get_stream_info(logged_in_uid, area_name):
        return jsonify({"message": "Stream inactive."}), 404

    STREAM_SOURCES[logged_in_uid][area_name]['line_coords'] = line_coords
    return jsonify({"message": "AOI set."}), 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    logged_in_uid = get_logged_in_uid(request)
    if not logged_in_uid: return jsonify({"message": "Unauthorized."}), 401

    area_name = request.json.get("areaName")

    if get_stream_info(logged_in_uid, area_name):
        cleanup(logged_in_uid, area_name) 
        return jsonify({"message": "Stream stopped."}), 200

    return jsonify({"message": "No stream."}), 200

@app.route('/signout')
def signout():
    global STREAM_SOURCES
    logged_in_uid = get_logged_in_uid(request)

    if logged_in_uid in STREAM_SOURCES:
        for area in list(STREAM_SOURCES[logged_in_uid].keys()):
            cleanup(logged_in_uid, area)

    cv2.destroyAllWindows()
    session.pop('GUEST_LOGS', None)
    session.pop('VIEWING_UID', None)
    session.pop('user', None)
    response = make_response(redirect(url_for('signin')))
    response.set_cookie('server_jwt', '', expires=0)
    return response

@app.route('/log_area_count', methods=['POST'])
def log_area_count():
    logged_in_uid = get_logged_in_uid(request)
    token = get_auth_token(request)
    if not logged_in_uid:
        return jsonify({"message": "Unauthorized."}), 401
    data = request.get_json()
    if not all(k in data for k in ('area', 'count', 'time')):
        return jsonify({"message": "Missing data."}), 400

    entry = {'area': data['area'], 'count': int(data['count']), 'time': data['time']}
    
    if logged_in_uid == GUEST_UID:
        logs = session.get('GUEST_LOGS', [])
        logs.append(entry)
        if len(logs) > MAX_LOG_SIZE:
            logs.pop(0)
        session['GUEST_LOGS'] = logs
    else:
        if not token: 
             return jsonify({"message": "Auth token missing for registered user."}), 401
             
        try:
            db.child(USER_LOGS_REF).child(logged_in_uid).child(data['area']).push(entry, token)
        except Exception as e:
            print(f"Firebase Log error: {e}")
            return jsonify({"message": "Auth issue or DB write failed."}), 500
            
    return jsonify({"message": "Logged.", "area": data['area']}), 200

@app.route('/get_historical_log', methods=['GET'])
def get_historical_log():
    display_uid, token, _ = get_display_uid_and_token(request)
    if not display_uid:
        return jsonify({"message": "Unauthorized."}), 410

    logs = []
    if display_uid == GUEST_UID:
        logs = session.get('GUEST_LOGS', [])
    else:
        all_logs = fetch_data_from_db(f"{USER_LOGS_REF}/{display_uid}", token) or {}
            
        logs = [entry for area in all_logs.values() if isinstance(area, dict) for entry in area.values()]

    if not logs:
        return jsonify({"message": "No data."}), 204

    five_min_ago = datetime.now(UTC) - timedelta(minutes=5)
    recent = []
    for e in logs:
        try:
            if datetime.fromisoformat(e['time'].replace('Z', '+00:00')) > five_min_ago:
                recent.append(e)
        except ValueError:
            continue
            
    return jsonify(recent)

@app.route('/get_server_chart_data', methods=['GET'])
def get_server_chart_data():
    h = generate_crowd_chart_image()
    d = generate_density_chart_image()
    return jsonify({"historical_chart_data": h, "density_chart_data": d}) if h or d else (jsonify({"message": "No data."}), 204)

@app.route('/api/camera_config', methods=['GET', 'POST'])
def api_camera_config():
    logged_in_uid = get_logged_in_uid(request)
    token = get_auth_token(request)

    if not logged_in_uid or logged_in_uid == GUEST_UID or not token:
        if request.method == 'GET':
            return jsonify({"configs": []}), 200
        return jsonify({"message": "Unauthorized."}), 401

    if request.method == 'GET':
        data = fetch_data_from_db(f"{USER_CAMERAS_REF}/{logged_in_uid}", token) or {}
        configs = list(data.values()) if isinstance(data, dict) else []
        return jsonify({"configs": configs}), 200

    # POST logic requires a token (registered user)
    try:
        payload = request.get_json(silent=True) or []
        if not isinstance(payload, list):
            return jsonify({"message": "Invalid payload"}), 400

        to_store = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            cam_id = item.get("id") or item.get("areaName", f"cam_{int(time.time() * 1000)}")
            to_store[cam_id] = item

        db.child(USER_CAMERAS_REF).child(logged_in_uid).set(to_store, token)
        return jsonify({"message": "Saved", "count": len(to_store)}), 200

    except Exception as e:
        print(f"Camera config error: {e}")
        return jsonify({"message": "DB error."}), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True)