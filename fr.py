import cv2
import dlib
import os

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Function to recognize faces in an image
# Function to recognize faces in an image
# Function to recognize faces in an image
def recognize_faces(image, detector, face_recognizer):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_encodings = []
    for face in faces:
        # Compute face descriptor for each face
        shape = predictor(gray, face)
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        face_encodings.append(face_descriptor)
    return face_encodings



# Function to compare face encodings
import numpy as np

# Function to compare face encodings
def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.6):
    # Calculate Euclidean distance between face encodings using NumPy
    distances = np.linalg.norm(np.array(known_face_encodings) - np.array(face_encoding_to_check), axis=1)
    return [distance <= tolerance for distance in distances]


# Folder containing images
folder_path = r"C:\Users\kitfc\Documents\GitHub\Face-recognition-python-project\Database"

# Load the selfie image
selfie_path = r"C:\Users\kitfc\Documents\GitHub\Face-recognition-python-project\persons\selfie.jpg"
selfie_image = cv2.imread(selfie_path)

# Initialize dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Initialize dlib's shape predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)


# Recognize faces in the selfie
selfie_face_encodings = recognize_faces(selfie_image, detector, face_recognizer)

# Load all images from the folder
all_images = load_images_from_folder(folder_path)

# Collect images where the person from the selfie appears
images_with_person = []
for img in all_images:
    face_encodings = recognize_faces(img, detector, face_recognizer)
    for face_encoding in face_encodings:
        if any(compare_faces(selfie_face_encodings, face_encoding)):
            images_with_person.append(img)
            break  # Once the person is detected, no need to check further in the same image

# Display or process the collected images
for img in images_with_person:
    cv2.imshow("Image with person", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
