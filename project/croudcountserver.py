from flask import Flask, render_template, redirect, url_for

app = Flask(__name__)

@app.route('/')
def root():
    return redirect(url_for('signin'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signout')
def signout():
    # In a real app, you would clear the user's session here.
    return redirect(url_for('signin'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/vid_analy')
def vid_analy():
    return render_template('vid_analy.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/cam_manage')
def cam_manage():
    return render_template('cam_manage.html')

if __name__ == '__main__':
    app.run(debug=True)