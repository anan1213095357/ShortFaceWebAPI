import os
import pickle
import numpy as np
import face_recognition
from flask import Flask, jsonify, request, redirect

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

# Path to the file where we'll store face encodings
KNOWN_FACES_FILE = 'known_faces.dat'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_known_faces():
    """Load known faces from disk."""
    if os.path.exists(KNOWN_FACES_FILE):
        with open(KNOWN_FACES_FILE, 'rb') as f:
            known_faces = pickle.load(f)
    else:
        known_faces = []
    return known_faces

def save_known_faces(known_faces):
    """Save known faces to disk."""
    with open(KNOWN_FACES_FILE, 'wb') as f:
        pickle.dump(known_faces, f)

# Load known faces at startup
known_faces = load_known_faces()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Check if both 'file' and 'name' are part of the POST request
        if 'file' not in request.files or 'name' not in request.form:
            return jsonify({"error": "Please provide a file and name"}), 400

        file = request.files['file']
        name = request.form['name']
        # Optional additional information
        age = request.form.get('age')
        email = request.form.get('email')

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # Load the image file
            img = face_recognition.load_image_file(file)
            # Get face encodings
            face_encodings = face_recognition.face_encodings(img)

            if len(face_encodings) == 0:
                return jsonify({"error": "No face found in the image"}), 400

            # Assuming one face per image
            face_encoding = face_encodings[0]

            # Save the encoding with user info
            known_faces.append({
                "name": name,
                "encoding": face_encoding.tolist(),
                "age": age,
                "email": email
            })

            save_known_faces(known_faces)

            return jsonify({"message": f"User {name} registered successfully"}), 200
        else:
            return jsonify({"error": "Invalid file type"}), 400
    else:
        # HTML form for registration
        return '''
        <!doctype html>
        <title>Register Face</title>
        <h1>Register a new face</h1>
        <form method="POST" enctype="multipart/form-data">
          Name: <input type="text" name="name"><br><br>
          Age: <input type="text" name="age"><br><br>
          Email: <input type="text" name="email"><br><br>
          <input type="file" name="file"><br><br>
          <input type="submit" value="Register">
        </form>
        '''

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'POST':
        if not known_faces:
            return jsonify({"error": "No faces have been registered yet."}), 400

        if 'file' not in request.files:
            return jsonify({"error": "Please provide a file"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            # Load the uploaded image file
            img = face_recognition.load_image_file(file)
            # Get face encodings for faces in the uploaded image
            face_encodings = face_recognition.face_encodings(img)

            if len(face_encodings) == 0:
                return jsonify({"error": "No face found in the image"}), 400

            unknown_encoding = face_encodings[0]

            # Prepare known encodings and user info
            known_encodings = [np.array(face['encoding']) for face in known_faces]

            # Compare the uploaded face with known faces
            face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
            best_match_index = np.argmin(face_distances)
            match_threshold = 0.6  # Adjust this value as needed

            if face_distances[best_match_index] < match_threshold:
                matched_face = known_faces[best_match_index]
                result = {
                    "face_found_in_image": True,
                    "identity": matched_face['name'],
                    "age": matched_face.get('age'),
                    "email": matched_face.get('email')
                }
            else:
                result = {
                    "face_found_in_image": True,
                    "identity": "Unknown"
                }

            return jsonify(result)
        else:
            return jsonify({"error": "Invalid file type"}), 400
    else:
        # HTML form for recognition
        return '''
        <!doctype html>
        <title>Recognize Face</title>
        <h1>Upload a picture to recognize</h1>
        <form method="POST" enctype="multipart/form-data">
          <input type="file" name="file"><br><br>
          <input type="submit" value="Upload">
        </form>
        '''

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
