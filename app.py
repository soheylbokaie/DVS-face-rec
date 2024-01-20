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
import jwt
import json
import requests
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import psycopg2
import face_recognition
import os



conn = psycopg2.connect(
    database="face_rec",
    user="root",
    password="1",
    host="127.0.0.1",
    port="5432"
)
cursor = conn.cursor()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

import hashlib

def sha256_hash(message):
    # Create a new SHA-256 hash object
    sha256 = hashlib.sha256()

    # Update the hash object with the bytes of the message
    sha256.update(message.encode('utf-8'))

    # Get the hexadecimal representation of the hash
    hashed_message = sha256.hexdigest()

    return hashed_message

#print(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
app = Flask(__name__)
app.config['SECRET_KEY'] = '7Uqpfps3M6EfTl5QxnsYtFRDQQAgcU42iI7H5eHAcOWHlhoyf1DhRFUeltfiXbQWsyQ9Sl5YVJbe2SdgDdO5qiPDT7b1S8rQ0'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
socketio = SocketIO(app,cors_allowed_origins="*")
CORS(app,cors_allowed_origins="*")  

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/signup')
def signup():
    return "render_template('teacher.html')"

@app.route('/generate_token',methods=['POST'])
def ss():
        data = request.get_json()
        data['user_to_enc']['face_id_verified'] = True
        token = jwt.encode(data, app.config['SECRET_KEY'], algorithm='HS256')
        url = 'http://172.20.10.5:8080/users/verify-face-signup'
        to_send = {"token":token}
        response = requests.post(url, json=to_send)

        return {'token':token}
    

@app.route('/save_pic',methods=['POST'])
def save_pic():
    global cursor
    global conn
    # try:
    data = request.get_json()
    decoded_data = base64.b64decode(data['image'])
    image_array = np.frombuffer(decoded_data, np.uint8)
    cv2_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    face_encoding = face_recognition.face_encodings(cv2_image)
    
    cursor.execute("INSERT INTO face_encodings (user_id, encoding) VALUES (%s, %s)", (data['user_id'], face_encoding[0].tobytes()))
    conn.commit()
    return {'status':"True"}


@app.route('/verf',methods=['POST'])
def verfPicture():
    global cursor
    global conn
    # try:
    data = request.get_json()
    election_id = data['election_id']
    decoded_data = base64.b64decode(data['image'])
    image_array = np.frombuffer(decoded_data, np.uint8)
    cv2_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    face_encoding = face_recognition.face_encodings(cv2_image)[0]
    cursor.execute(f"select * from face_encodings where user_id = {data['user_id']} ")
    rows = cursor.fetchone()
    
    face_to_check  = np.frombuffer(bytes(rows[1]), dtype=np.float64)

    results = face_recognition.compare_faces([face_to_check], face_encoding)[0]
    text_to_encrypt = f"{data['user_id']} {election_id} {results}"
    encrypted_text = sha256_hash(text_to_encrypt)
    print(encrypted_text)
    return {"hash": str(encrypted_text)}



@socketio.on('connect')
def handle_connect():
    print('WebSocket client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('WebSocket client disconnected')

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
])


def process_image(image,session_id,state,name=None):
    image_array = np.frombuffer(image, np.uint8)
    cv2_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    size = cv2_image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )



    faces = detector(gray)
    direction = "Front"
 
    if len(faces) > 0 :
        face = faces[0]
        y = face.top()
        b = face.bottom()
        l = face.left()
        r = face.right()
        face_for_teaching = cv2_image[ y:b,l:r]
        landmarks = predictor(gray, face)


        color = (0, 0, 255)
        image_points = []
        
        for i in [30,8,36,45,48,54]:
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            image_points.append((x,y)) 
            
            # cv2.circle(cv2_image, (x, y), 3, color, -1)
        image_points = np.array(image_points, dtype="double")


        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
        # for p in image_points:
            # cv2.circle(cv2_image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        # cv2.line(cv2_image, p1, p2, (255,0,0), 2)


        if abs(p2[0] - p1[0] ) > (size[0]/3) and (p2[0] < p1[0]):
            direction = 'Right'
            # cv2.imwrite('./face/'+direction+'.jpg',face_for_teaching)

        elif abs(p2[0] - p1[0] ) > (size[0]/3) and (p2[0] > p1[0]):
            direction = 'Left'
            # cv2.imwrite('./face/'+direction+'.jpg',face_for_teaching)


            
            # cv2.imwrite('./face/'+direction+'.jpg',face_for_teaching)
# 

    _, buffer = cv2.imencode('.jpg', cv2_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    
    data_to_emit = {"direction":direction}
    if state == 6:
        data_to_emit['image'] = encoded_image
    socketio.emit('data',data_to_emit,room=session_id)
    

def teach(name,face):
    print("start")
    labels = [name]*4
    faces = []
    loaded_knn_classifier = joblib.load('./knn_model.pkl')

    for i in face :
        i = cv2.resize(i, (100,100))
        i = i.flatten()  
        faces.append(i)
    
    # faces = faces.reshape(-1, -1)
    loaded_knn_classifier.fit(faces, labels)
    model_filename = 'knn_model.pkl'
    joblib.dump(loaded_knn_classifier, model_filename)

def predict(face):
    face = cv2.resize(face, (100,100))
    face = face.flatten()
    loaded_knn_classifier = joblib.load('./knn_model.pkl')
    name = loaded_knn_classifier.predict(face)
    return name








@socketio.on('get_image')
def handle_custom_event(data):
    session_id = request.sid
    process_image(base64.b64decode(data['img'].split(',')[1]),session_id,name=data['name'],state=data['state'])

if __name__ == '__main__':
    socketio.run(app,ssl_context=('../face_rec/cert.pem', '../face_rec/key.pem'),host='0.0.0.0')
    # socketio.run(app)

cursor.close()
conn.close()
