from flask import Flask, request, Response
import cv2
from video_gen import process_video_realtime

app = Flask(__name__)

@app.route('/')
def root():
    return app.send_static_file('./index.html')

@app.route('/video_stream')
def video_stream():
    filename = request.args.get("filename")
    type = request.args.get("type")
    print(filename,type)
    if type == 'v':
        return Response(process_video_realtime(filename),
            mimetype='multipart/x-mixed-replace; boundary=frame')
    elif type == 'w':
        return Response(process_video_realtime(0),
            mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)  # Adjust host and port as needed