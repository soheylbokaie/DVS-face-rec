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

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
])


def process_image(image,session_id,name=None):
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
    direction = "front"
 
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
            direction = 'right'
            cv2.imwrite('./face/'+direction+'.jpg',face_for_teaching)

        elif abs(p2[0] - p1[0] ) > (size[0]/3) and (p2[0] > p1[0]):
            direction = 'left'
            cv2.imwrite('./face/'+direction+'.jpg',face_for_teaching)

        elif abs(p2[1] - p1[1] ) > (size[1]/5) and (p2[1] < p1[1]):
            direction = 'up'
            cv2.imwrite('./face/'+direction+'.jpg',face_for_teaching)
        else:
            cv2.imwrite('./face/'+direction+'.jpg',face_for_teaching)


    _, buffer = cv2.imencode('.jpg', cv2_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    
    
    
    socketio.emit('data',{'img':encoded_image,"direction":direction
                          },room=session_id)
    

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
    process_image(base64.b64decode(data['img'].split(',')[1]),session_id,name=data['name'])

if __name__ == '__main__':
    socketio.run(app,ssl_context=('./cert.pem', './key.pem'),host='0.0.0.0')
    # socketio.run(app)

