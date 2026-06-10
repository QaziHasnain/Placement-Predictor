import os
import sys

import joblib
import pandas as pd
from flask import Flask, redirect, render_template, request, Response
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_pymongo import PyMongo
from werkzeug.security import check_password_hash, generate_password_hash


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(BASE_DIR, 'src')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'placement_model.pkl')
COLUMNS_PATH = os.path.join(BASE_DIR, 'models', 'model_columns.pkl')

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
app.config['MONGO_URI'] = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/placementDB')

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

mongo = PyMongo(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.role = user_data.get('role', 'student')


@login_manager.user_loader
def load_user(user_id):
    from bson.objectid import ObjectId

    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        return User(user)
    return None


def password_is_valid(stored_password, submitted_password):
    if stored_password.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(stored_password, submitted_password)
    return stored_password == submitted_password


def placement_probability(input_df):
    probability = float(model.predict_proba(input_df)[0][1])
    if probability > 0.95:
        probability = 0.92
    return probability


def default_prediction_form():
    return {
        'age': '21',
        'gender': 'Male',
        'degree': 'B.Tech',
        'branch': 'CSE',
        'cgpa': '8.5',
        'internships': '1',
        'projects': '2',
        'coding': '7',
        'comm': '7',
        'aptitude': '70',
        'soft_skills': '7',
        'certs': '1',
        'backlogs': '0',
    }


@app.route('/')
def home():
    return redirect('/login')


@app.route('/favicon.ico')
def favicon():
    return Response(status=204)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password']

        if mongo.db.users.find_one({"username": username}):
            return "Username already exists", 409

        mongo.db.users.insert_one({
            "username": username,
            "email": email,
            "password": generate_password_hash(password),
            "role": "student",
        })
        return redirect('/login')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        user_data = mongo.db.users.find_one({"username": username})

        if user_data and password_is_valid(user_data.get('password', ''), password):
            user = User(user_data)
            login_user(user)
            return redirect('/dashboard')

        return "Invalid credentials", 401
    return render_template('login.html')


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)


@app.route('/admin')
@login_required
def admin():
    if current_user.role != 'admin':
        return "Access denied", 403
    return "Welcome Admin Panel"


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    result = None
    probability = None
    form_data = default_prediction_form()

    if request.method == 'POST':
        form_data.update(request.form.to_dict())
        input_df = pd.DataFrame([{
            'Age': int(form_data['age']),
            'Gender': form_data['gender'],
            'Degree': form_data['degree'],
            'Branch': form_data['branch'],
            'CGPA': float(form_data['cgpa']),
            'Internships': int(form_data['internships']),
            'Projects': int(form_data['projects']),
            'Coding_Skills': int(form_data['coding']),
            'Communication_Skills': int(form_data['comm']),
            'Aptitude_Test_Score': int(form_data['aptitude']),
            'Soft_Skills_Rating': int(form_data['soft_skills']),
            'Certifications': int(form_data['certs']),
            'Backlogs': int(form_data['backlogs']),
        }], columns=model_columns)

        prob = placement_probability(input_df)
        result = 'PLACED' if prob >= 0.5 else 'NOT PLACED'
        probability = round(prob * 100, 1)

    return render_template(
        'predict.html',
        result=result,
        probability=probability,
        form_data=form_data,
    )


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
