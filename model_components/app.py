from flask import Flask, request, Response, render_template
from LD import process_video_realtime


app = Flask(__name__, static_url_path='/static', static_folder='static')

@app.route('/')
def index():
    return render_template('../index.html')

@app.route('/video_stream')
def video_stream():
    filename = request.args.get("filename")
    type = request.args.get("type")
    print(filename,type)
    if type == 'w':
        filename = 0
    return Response(process_video_realtime(filename),
            mimetype='multipart/x-mixed-replace; boundary=frame')
        

if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000)  # Adjust host and port as needed