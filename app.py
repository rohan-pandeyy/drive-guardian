from flask import Flask, request, Response
import cv2
from video_gen import process_video_realtime

app = Flask(__name__)

@app.route('/')
def root():
    return app.send_static_file('./index.html')

@app.route('/video_stream')
def video_stream():
    return Response(process_video_realtime('project_video.mp4'),
        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)  # Adjust host and port as needed