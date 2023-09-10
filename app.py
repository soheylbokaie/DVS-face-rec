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
# import mediapipe as mp

# Initialize MediaPipe Face Detection and Facial Landmark modules

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
# lips_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')

# mp_face_detection = mp.solutions.face_detection
# mp_face_mesh = mp.solutions.face_mesh
# mp_drawing = mp.solutions.drawing_utils

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
    rgb_frame = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    #     results = face_mesh.process(rgb_frame)
    #     print(results)

        
    
    
    faces = face_cascade.detectMultiScale(gray,1.7,1)
    
    for (x, y, w, h) in faces:
        face = gray[ y:y+h,x:x+w]
        
        if (gray.size) > 0 :
            eyes = eyes_cascade.detectMultiScale(face)
            # noses = nose_cascade.detectMultiScale(face)
            face = cv2.resize(face, (100, 100))
            face = face.reshape(1, -1) # Flatten to 1D array
            try:
                result = loaded_knn_classifier.predict(face)
                eyes_df = pd.DataFrame(eyes,columns=['x','y','w','h'])
                eyes_df['l'] = eyes_df['w']*eyes_df['h']
                eyes_df.sort_values(by=['l'],inplace=True)
                # s.to_csv('./soheyl.csv')
                # real_eyes = [(x1,y1,w1,h1) for (x1,y1,w1,h1) in eye]
                # max_eyin real_eyes:
                eyes = eyes_df.values[0:2]
                eye_1 = eyes[0]
                eye_2 = eyes[1]
                if eye_1[0] > eye_2[0]:
                    left_eye = eye_2
                    right_eye = eye_1
                else:
                    left_eye = eye_1
                    right_eye = eye_2
                
                # for eye in eyes_df.iloc[:2].values :
                cv2.rectangle(cv2_image, (x+left_eye[0], y+left_eye[1]), (x+left_eye[0]+left_eye[2], y+left_eye[1]+left_eye[3]), (0, 255, 255), 2)
                cv2.putText(cv2_image, str('right'), (x+left_eye[0],y+left_eye[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(cv2_image, (x+right_eye[0], y+right_eye[1]), (x+right_eye[0]+right_eye[2], y+right_eye[1]+right_eye[3]), (0, 255, 255), 2)
                cv2.putText(cv2_image, str('left'), (x+right_eye[0],y+right_eye[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except:
                pass
            # for nose in noses:
                # cv2.rectangle(cv2_image, (x+nose[0], y+nose[1]), (x+nose[0]+nose[2], y+nose[1]+nose[3]), (0, 0, 255), 2)
                
            
                            
            cv2.rectangle(cv2_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # cv2.putText(cv2_image, str(result[0]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
    _, buffer = cv2.imencode('.jpg', cv2_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    
    
    socketio.emit('data',encoded_image,room=session_id)
    




@socketio.on('get_image')
def handle_custom_event(data):
    session_id = request.sid
    process_image(base64.b64decode(data.split(',')[1]),session_id)

if __name__ == '__main__':
    socketio.run(app,ssl_context=('/usr/local/etc/nginx/server.crt', '/usr/local/etc/nginx/server.key'))
    # socketio.run(app)

