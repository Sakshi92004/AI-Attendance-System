pip install opencv-python
pip install face_recognition
pip install numpy
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
import csv

# Load known faces
known_face_encodings = []
known_face_names = []

face_dir = "known_faces"

for filename in os.listdir(face_dir):
    if filename.endswith("image.jpg"):
        image = face_recognition.load_image_file(f"{face_dir}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
        known_face_names.append(os.path.splitext(filename)[0])

# Initialize video capture
video_capture = cv2.VideoCapture(0)

students_marked = []

# Create/open CSV to log attendance
now = datetime.now()
date_string = now.strftime("%Y-%m-%d")

with open("attendance.csv", "a", newline="") as file:
    writer = csv.writer(file)
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                if name not in students_marked:
                    students_marked.append(name)
                    time_string = now.strftime("%H:%M:%S")
                    writer.writerow([name, date_string, time_string])
                    print(f"{name} marked present at {time_string}")

            # Draw rectangle around face
            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

video_capture.release()
cv2.destroyAllWindows()
