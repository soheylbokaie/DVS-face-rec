from flask import Flask,render_template,request, Response
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
from flask_cors import CORS
import time
import random
import joblib
import base64
import pandas as pd
import dlib

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
# eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#print(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
loaded_knn_classifier = joblib.load('./knn_model.pkl')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
socketio = SocketIO(app,cors_allowed_origins="*")
CORS(app,cors_allowed_origins="*")  

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/signup')
def signup():
    return render_template('teacher.html')


@socketio.on('connect')
def handle_connect():
    print('WebSocket client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('WebSocket client disconnected')


def process_image(image,session_id):
    image_array = np.frombuffer(image, np.uint8)
    cv2_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)





    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)


        for i in [8,36,39,42,45,30,48,54]:
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            color = (0, 0, 255)
            
            cv2.circle(cv2_image, (x, y), 3, color, -1)
        
    _, buffer = cv2.imencode('.jpg', cv2_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    
    
    socketio.emit('data',{'img':encoded_image,
                          },room=session_id)
    




@socketio.on('get_image')
def handle_custom_event(data):
    session_id = request.sid
    process_image(base64.b64decode(data.split(',')[1]),session_id)

if __name__ == '__main__':
    socketio.run(app,ssl_context=('../cert.pem', '../key.pem'),host='0.0.0.0')
    # socketio.run(app)

