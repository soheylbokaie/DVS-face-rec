
from flask import Flask, Response, request
import cv2
import base64

app = Flask(__name__)

# Function to capture image from webcam using OpenCV
def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

@app.route('/get_image', methods=['GET'])
def get_image():
    frame = capture_frame()
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

if __name__ == '__main__':
    app.run(debug=True)
