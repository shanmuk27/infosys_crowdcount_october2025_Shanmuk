from flask import Flask, render_template, redirect, url_for, request, session, make_response
import datetime
import pyrebase
import jwt

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

app = Flask(__name__)
app.secret_key = 'a-very-secret-key'  # used for session and JWT signing
JWT_SECRET = 'another-very-secret-key'  # server JWT secret
JWT_ALGORITHM = 'HS256'

def get_user_data(uid, id_token):
    """Get user data from Firebase DB"""
    return db.child("user").child(uid).get(id_token).val()

def generate_server_jwt(uid, email, username):
    """Generate a JWT token for app session"""
    current_time = datetime.datetime.utcnow()
    payload = {
        "uid": uid,
        "email": email,
        "username": username,
        "iat": current_time,
        "exp": current_time + datetime.timedelta(hours=1)  # expires in 1 hour
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
        
        # Firebase login
        user = auth.sign_in_with_email_and_password(email, password)
        uid = user['localId']
        id_token = user['idToken']

        # Fetch user data from Firebase
        user_data = get_user_data(uid, id_token)
        username = user_data.get('user_name', 'User')

        # Create server JWT
        server_jwt = generate_server_jwt(uid, email, username)

        # Store session info (optional)
        session['user'] = {'uid': uid, 'idToken': id_token, 'jwt': server_jwt}

        # Optionally, send JWT in cookie
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
    jwt_token = request.cookies.get('server_jwt')
    decoded = verify_server_jwt(jwt_token)
    if not decoded:
        return redirect(url_for('signin'))
    
    username = decoded.get('username', 'User')
    return render_template('index.html', username=username)

@app.route('/profile')
def profile():
    jwt_token = request.cookies.get('server_jwt')
    decoded = verify_server_jwt(jwt_token)
    if not decoded:
        return redirect(url_for('signin'))

    username = decoded.get('username', 'User')
    email = decoded.get('email', '')
    return render_template('profile.html', username=username, email=email)

@app.route('/home')
def home():
    jwt_token = request.cookies.get('server_jwt')
    decoded = verify_server_jwt(jwt_token)
    if not decoded:
        return redirect(url_for('signin'))
    
    return render_template('home.html')

@app.route('/signout')
def signout():
    session.pop('user', None)
    response = make_response(redirect(url_for('signin')))
    response.set_cookie('server_jwt', '', expires=0)  # remove JWT cookie
    return response

if __name__ == '__main__':
    app.run(debug=True)
