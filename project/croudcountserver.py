from flask import Flask, render_template, redirect, url_for, request, session
import pyrebase

#for firebase
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

firebase=pyrebase.initialize_app(config)
auth=firebase.auth()
db=firebase.database()

app = Flask(__name__)
app.secret_key = 'a-very-secret-key'

@app.route('/')
def root():
    return redirect(url_for('signin'))

@app.route('/index')
def index():
    if 'user' in session:
        uid = session['user']['uid']
        idTOken=session['user']['idToken']
        user_data = db.child("user").child(uid).get(idTOken).val()
        username = user_data.get('user_name', 'User')
        return render_template('index.html', username=username)
    else:
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
        session['user'] ={'uid': user['localId'],'idToken': user['idToken']}
        return redirect(url_for('index'))
    except Exception as e:
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
        error = "Could not register. The email might already be in use."
        return render_template('signin.html', Error=error,mode='register')

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('signin'))
    return render_template('profile.html')

@app.route('/signout')
def signout():
    session.pop('user', None)
    return redirect(url_for('signin'))

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('signin'))
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)