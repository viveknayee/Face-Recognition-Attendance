from flask import Flask,render_template,session,redirect,request,Response
from models import db,Attendance
import os
from datetime import date,datetime
from flask import flash
import threading  
import cv2
from better_face_rec import build_or_update_records, build_index_from_records, load_records, recognize_frame                                  
import csv
import io
from flask import make_response

IMAGES_DIR = "images"
attendance_status = {}

records = load_records()
enc_index, names_index, paths_index = build_index_from_records(records)

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

app.secret_key = "faceattend123"

users = {
    "admin": os.environ.get("ADMIN_PASSWORD", "Face@2026")
}

@app.route("/",methods = ['GET','POST'])
def login():
    if request.method == 'POST':
        username  = request.form['username']
        password  = request.form['password']

        if username in users and users[username]== password:
            session['user'] = username
            return redirect("/dashboard")
        else:
            error = "Wrong username or password"
            return render_template("login.html", error=error)
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    username = session["user"]
    today = date.today()
    
    present_today = Attendance.query.filter_by(date=today).count()
    registered_faces = len(os.listdir(IMAGES_DIR)) if os.path.exists(IMAGES_DIR) else 0
    today_str = today.strftime("%d %b %Y")
    
    return render_template("dashboard.html", 
                         username=username, 
                         today=today_str,
                         present_today=present_today,
                         registered_faces=registered_faces)


@app.route("/logout")
def logout():
    session.pop("user")
    return redirect("/")    
with app.app_context():
    db.create_all()

@app.route("/register", methods=['GET','POST'])
def register():

    if "user" not in session:
        return redirect("/")
    
    if request.method == 'POST':
        name = request.form["name"]
        photos = request.files.getlist("photo")

        if photos:
            person_dir = os.path.join(IMAGES_DIR, name)
            os.makedirs(person_dir, exist_ok=True)
            for photo in photos:
                photo.save(os.path.join(person_dir, photo.filename))
            thread = threading.Thread(target=build_or_update_records)
            thread.start()
            flash(f"{name} registered successfully!", "success")
            return redirect("/dashboard")
        else:
            error = "Please upload a photo!"
            return render_template("register.html", error=error)
        
    return render_template("register.html")

    if "user" not in session:
        return redirect("/")
    
    if request.method == 'POST':
        name = request.form["name"]
        photo = request.files["photo"]

        if photo:
            person_dir = os.path.join(IMAGES_DIR, name)
            os.makedirs(person_dir, exist_ok=True)
            photo.save(os.path.join(person_dir,photo.filename))
            thread = threading.Thread(target=build_or_update_records)
            thread.start()
            flash(f"{name} registered successfully!", "success")
            return redirect("/dashboard")
        else:
            error = "Please upload a photo!"
            return render_template("register.html",error=error)
        
    return render_template("register.html")


def generate_frames():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            name = "Unknown"
            dist = None

            results = recognize_frame(frame, enc_index, names_index)
            
            for (top, right, bottom, left, name, dist) in results:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                if dist is not None:
                    confidence = round((1 - dist) * 100)
                    label = f"{name} {confidence}%"
                else:
                    label = name
                cv2.putText(frame, label, (left+6, bottom-6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            if name != "Unknown":
                with app.app_context():
                    today = date.today()
                    existing = Attendance.query.filter_by(
                        name=name, date=today).first()
                    if not existing:
                        new_attendance = Attendance(
                            name=name,
                            date=today,
                            time=datetime.now().time(),
                            confidence=round((1-dist)*100) if dist else None
                        )
                        db.session.add(new_attendance)
                        db.session.commit()
                        attendance_status[name] = "marked"
                    else:
                        attendance_status[name] = "already"
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/attendance')
def attendance():
    if "user" not in session:
        return redirect("/")
    records = Attendance.query.order_by(Attendance.date.desc()).all()
    return render_template('attendance.html', records=records)

@app.route('/live')
def live():
    if "user" not in session:
        return redirect("/")
    attendance_status.clear()
    return render_template('live.html')


@app.route('/attendance_status')
def get_attendance_status():
    from flask import jsonify
    return jsonify(attendance_status)
 
@app.route('/export_csv')
def export_csv():
    if "user" not in session:
        return redirect("/")
    
    records = Attendance.query.order_by(Attendance.date.desc()).all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Date', 'Time', 'Confidence'])
    
    for record in records:
        writer.writerow([record.name, record.date, record.time, record.confidence])
    
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=attendance.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

if __name__ == "__main__":
    app.run(debug=True)