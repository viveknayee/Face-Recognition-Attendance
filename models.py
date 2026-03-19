from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(100),nullable = False)
    date = db.Column(db.Date, default = datetime.now().date)
    time = db.Column(db.Time, default = datetime.now().time)
    confidence = db.Column(db.Float, nullable = True)

    def __repr__(self):
        return f"{self.id} - {self.name}"