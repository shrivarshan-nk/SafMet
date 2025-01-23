import cv2
import numpy as np
import os
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import time

# Initialize directories
if not os.path.exists('enrollment_data'):
    os.mkdir('enrollment_data')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Model file path
MODEL_PATH = 'model2.h5'

image_dir = 'C:/Users/shriv/Biometrics/Dataset'  # Modify this as needed

# Create a label map from the directory structure
label_map = {}
for idx, folder in enumerate(os.listdir(image_dir)):
    folder_path = os.path.join(image_dir, folder)
    if os.path.isdir(folder_path):  # Check if it's a directory
        label_map[idx] = folder  # Map index to folder (celebrity name)
        
        
# YOLO Model (Assuming you have a YOLO model for helmet detection)
yolo_net = cv2.dnn.readNetFromDarknet('yolov3-helmet.cfg', 'yolov3-helmet.weights')
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# CNN model definition
def build_cnn_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the model if it exists, otherwise create a new one
def load_or_create_model(num_classes):
    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model = load_model(MODEL_PATH)
    else:
        print("No saved model found. Creating new model...")
        model = build_cnn_model(num_classes)
    return model

# Preprocess the image (grayscale conversion, equalization, and denoising)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    enhance = enhance_with_laplacian(equalized)
    return enhance

def enhance_with_laplacian(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened
# Detect face and return cropped image
def detect_and_crop_face(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        if w > 0 and h > 0:  # Check for valid dimensions
            return image[y:y+h, x:x+w], (x, y, w, h)
    return None, None  # No face detected

# Enroll user and add image to dataset
def enroll_user(user_id, frame, model, training_data, labels):
    face_img, _ = detect_and_crop_face(frame)
    if face_img is not None:
        face_img = preprocess_image(face_img)
        resized_img = cv2.resize(face_img, (64, 64))
        training_data.append(resized_img)
        labels.append(int(user_id))
        cv2.imwrite(f"enrollment_data/user_{user_id}.png", resized_img)
        messagebox.showinfo("Enrollment", f"User {user_id} enrolled successfully!")
        print(f"Enrollment data count: {len(training_data)}, Labels count: {len(labels)}")
    else:
        messagebox.showerror("Enrollment Error", "No face detected. Please try again.")

# Train the model with sufficient samples and update the model
def train_and_update_model(model, training_data, labels):
    if len(training_data) < 2 or len(labels) < 2:
        print("Not enough data to train the model.")
        return
    training_data = np.array(training_data).reshape(-1, 64, 64, 1).astype('float32') / 255
    labels = to_categorical(labels, num_classes=10)
    print(f"Training with {len(training_data)} samples and {len(labels)} labels.")
    model.fit(training_data, labels, epochs=10, batch_size=5, verbose=1)
    print("Model training complete.")
    model.save(MODEL_PATH)  # Save the updated model
    print(f"Model saved to {MODEL_PATH}")
    
# Identify user using the CNN model
def identify_user(model, face_img):
    if face_img is not None and face_img.size > 0:
        face_img = preprocess_image(face_img)
        resized_img = cv2.resize(face_img, (64, 64)).reshape(1, 64, 64, 1).astype('float32') / 255
        predictions = model.predict(resized_img)
        user_id = np.argmax(predictions)
        return label_map[user_id] if np.max(predictions) > 0.6 else None
    else:
        print("Error: face_img is empty or invalid.")
        return None

# Perform YOLO-based helmet detection and draw bounding box
def detect_helmet(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)
    
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming class 0 is helmet in your YOLO model
                # Draw bounding box
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])
                x1 = int(center_x - width / 2)
                y1 = int(center_y - height / 2)
                x2 = x1 + width
                y2 = y1 + height

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                return True
    return False

verified=0


def is_frame_too_dark(frame, threshold=50):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray_frame)
    return avg_brightness < threshold

# Detect occlusions by checking face dimensions
def is_face_occluded(face_img, min_dimension=30):
    h, w = face_img.shape[:2]
    return h < min_dimension or w < min_dimension

# Updated capture_frame function
def capture_frame(window, is_enrollment, model, training_data, labels):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open video feed.")
        return
# Capture frame for enrollment or identification
def capture_frame(window, is_enrollment, model, training_data, labels):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to open video feed.")
        return

    user_id_last_detected = None
    detection_start_time = None
    helmet_detected = False

    def update_frame():
        nonlocal user_id_last_detected, detection_start_time, helmet_detected
        global verified

        ret, frame = cap.read()
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_frame)
            if (verified==1):
                
                # Run YOLO detection for helmet
                helmet_detected = detect_helmet(frame)
                if helmet_detected:
                    print("Helmet detected")
                    cv2.putText(frame, "Helmet Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #messagebox.showinfo("Helmet Detection", "Helmet detected. Verification complete.")
                    #window.destroy()
                    #cap.release()
                    #return
                else:
                    print("Please wear a helmet warning triggered")
                    cv2.putText(frame, "Please wear a helmet!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
            

            elif brightness < 50:  # Adjust threshold for brightness as needed
                cv2.putText(frame, "Low light detected! Improve lighting.", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:    

                # Detect faces
                faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
                if len(faces) == 0:
                    cv2.putText(frame, "No face detected! Check for occlusions.", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y + h, x:x + w]
                        if face_img is not None and face_img.size > 0:
                            if is_enrollment:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            else:
                                user_id = identify_user(model, face_img)
                
                                # Ensure consistency in user detection
                                if (user_id is not None)  :
                                    print(f"User ID detected: {user_id}")
                                    
                                    # Verify if detected ID matches the previous detection
                                    if user_id_last_detected == user_id and (verified==0) :
                                        # Start detection time or verify consistent detection for 3 seconds
                                        if detection_start_time is None:
                                            detection_start_time = time.time()
                                        elif time.time() - detection_start_time > 3:
                                            verified=1
                                            # Check for helmet only if not yet detected
                                    
                                    else:
                                        # Reset detection time if user ID changes
                                        detection_start_time = None
                                        user_id_last_detected = user_id
                                    
                                    # Display bounding box and user ID
                                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    cv2.putText(frame, f"ID: {user_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                else:
                                      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                      cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)       
            # Update the GUI with the frame
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            video_label.config(image=img_tk)
            video_label.image = img_tk
    
        if cap.isOpened():
            window.after(10, update_frame)


    def capture():
        ret, frame = cap.read()
        cap.release()
        if ret and is_enrollment:
            enroll_window = Toplevel(window)
            enroll_window.title("Enroll User")
            enroll_window.geometry("300x300")
            enroll_window.configure(bg="lightgreen") 
            Label(enroll_window, text="Enter User ID:").pack(pady=10)
            user_id_entry = Entry(enroll_window)
            user_id_entry.pack(pady=10)
            Label(enroll_window, text="Enter User Name:").pack(pady=10)
            user_name_entry = Entry(enroll_window)
            user_name_entry.pack(pady=10)
            
            def enroll_user_callback():
                user_id = user_id_entry.get()
                if user_id:
                    enroll_user(user_id, frame, model, training_data, labels)
                    train_and_update_model(model, training_data, labels)
                    enroll_window.destroy()
                    window.destroy()
            
            enroll_button = Button(enroll_window, text="Enroll", command=enroll_user_callback)
            enroll_button.pack(pady=10)
        else:
            window.destroy()

    # Setup window
    video_label = Label(window)
    video_label.pack()

    capture_button = Button(window, text="Capture Frame" if is_enrollment else "Close", command=capture)
    capture_button.pack(pady=10)

    update_frame()

# Enrollment and verification functions
def open_enrollment_window():
    enroll_window = Toplevel(root)
    enroll_window.title("Enroll User")
    capture_frame(enroll_window, is_enrollment=True, model=cnn_model, training_data=training_data, labels=labels)

def open_verification_window():
    global verified
    verified = 0
    verify_window = Toplevel(root)
    verify_window.title("Identify User")
    capture_frame(verify_window, is_enrollment=False, model=cnn_model, training_data=training_data, labels=labels)

# Main Tkinter GUI setup
root = Tk()
root.title("Biometric Helmet Detection System")
root.geometry("300x300")
root.configure(bg="lightgreen") 

# Initialize training data and labels
training_data = []
labels = []

# Load or create the model
cnn_model = load_or_create_model(10)  # Assuming 10 classes for users (adjust if needed)

label = Label(root, text="Facial and Helmet Detection!", font=("Arial", 16),bg="lightgreen")
label.pack(pady=10)  
# Buttons for enrollment and verification
enroll_button = Button(root, text="Enroll User", command=open_enrollment_window)
enroll_button.pack(pady=75)

verify_button = Button(root, text="Verify User", command=open_verification_window)
verify_button.pack(pady=10)

root.mainloop()
