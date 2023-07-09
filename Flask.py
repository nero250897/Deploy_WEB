from flask import Flask, Response

# Required to run the YOLOv8 model
import cv2

from YOLO_Video import video_detection
app = Flask(__name__)

app.config['SECRET_KEY'] = 'DangKhanhHuy'
app.config['UPLOAD_FOLDER'] = 'static/files'

def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)

        frame=buffer.tobytes()
        yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')


# To display the Output Video on Webcam page
@app.route('/webcam')
def webapp():
    return Response(generate_frames_web(path_x=1), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)