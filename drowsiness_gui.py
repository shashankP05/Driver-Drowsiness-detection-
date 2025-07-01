import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import Label, Button, Frame, filedialog
from PIL import Image, ImageTk
import threading
import time

# Load drowsiness model (update path if needed)
drowsiness_model = load_model(r'C:\Users\shash\Desktop\drowsiness\final_eye_state_model_finetuned2.keras')

# Load age detection model
AGE_MODEL = 'age_net.caffemodel'
AGE_PROTO = 'age_deploy.prototxt'
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Eye and face detection (Haarcascade)
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Helper for age prediction
def predict_age(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    return AGE_LIST[age_preds[0].argmax()]

def predict_eye_state(eye_img):
    img = cv2.resize(eye_img, (224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = drowsiness_model.predict(img)
    return int(pred[0][0] > 0.5)

class DrowsinessApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Drowsiness & Age Detection')
        self.root.configure(bg='#f0f0f0')
        self.panel_frame = Frame(root, bd=3, relief='groove', bg='#e6e6e6')
        self.panel_frame.pack(padx=20, pady=20)
        self.panel = Label(self.panel_frame, bg='#e6e6e6')
        self.panel.pack()
        self.status = Label(root, text='', font=('Arial', 16, 'bold'), bg='#f0f0f0')
        self.status.pack(pady=(10,0))
        self.drowsy_count_label = Label(root, text='', font=('Arial', 14, 'bold'), bg='#f0f0f0', fg='blue')
        self.drowsy_count_label.pack(pady=(0,10))
        self.button_frame = Frame(root, bg='#f0f0f0')
        self.button_frame.pack(pady=(0,20))
        Button(self.button_frame, text='Start Webcam', width=15, command=self.start_webcam, bg='#4CAF50', fg='white', font=('Arial', 12, 'bold')).pack(side='left', padx=10)
        Button(self.button_frame, text='Upload Image', width=15, command=self.upload_image, bg='#2196F3', fg='white', font=('Arial', 12, 'bold')).pack(side='left', padx=10)
        self.stop_event = threading.Event()
        self.cap = None
        self.drowsy_counter = 0
        self.face_eye_closed_times = {}  # Track closed-eye start times per face
        self.drowsy_threshold_sec = 2    # Seconds eyes must be closed to be considered drowsy

    def start_webcam(self):
        self.stop_event.clear()
        self.cap = cv2.VideoCapture(0)
        threading.Thread(target=self.video_loop, daemon=True).start()

    def video_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break
            display_frame, drowsy, drowsy_people = self.process_frame_with_timer(frame)
            self.show_frame(display_frame)
            if drowsy:
                self.status.config(text='Drowsiness Detected!', fg='red')
            else:
                self.status.config(text='Normal', fg='green')
            self.drowsy_count_label.config(text=f'Drowsy People: {drowsy_people}')
        if self.cap:
            self.cap.release()

    def process_frame_with_timer(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
        drowsy_people = 0
        annotated_frame = frame.copy()
        now = time.time()
        drowsy_any = False
        new_face_eye_closed_times = {}
        prev_face_ids = list(self.face_eye_closed_times.keys())
        prev_face_centers = [(fid, ((fid[0]+fid[2]//2)*10, (fid[1]+fid[3]//2)*10)) for fid in prev_face_ids]
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            age = predict_age(face_img)
            eyes = EYE_CASCADE.detectMultiScale(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
            eye_states = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_img = face_img[ey:ey+eh, ex:ex+ew]
                state = predict_eye_state(eye_img)
                eye_states.append(state)
                color = (0,255,0) if state else (0,0,255)
                cv2.rectangle(annotated_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
            # Improved face matching: find closest previous face center
            face_center = (x + w//2, y + h//2)
            matched_id = None
            min_dist = 50  # pixels
            for fid, prev_center in prev_face_centers:
                dist = ((face_center[0] - prev_center[0])**2 + (face_center[1] - prev_center[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    matched_id = fid
            if matched_id is not None:
                closed_start = self.face_eye_closed_times[matched_id]
                face_id = matched_id
            else:
                closed_start = now
                face_id = (x//10, y//10, w//10, h//10)
            if len(eye_states) == 2 and sum(eye_states) == 0:
                closed_duration = now - closed_start
                new_face_eye_closed_times[face_id] = closed_start
                if closed_duration >= self.drowsy_threshold_sec:
                    drowsy_people += 1
                    drowsy_any = True
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0,0,255), 2)
                    label = f'Drowsy | Age: {age}'
                else:
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0,255,255), 2)
                    label = f'Blinking | Age: {age}'
            else:
                new_face_eye_closed_times[face_id] = now
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0,255,0), 2)
                label = f'Normal | Age: {age}'
            cv2.rectangle(annotated_frame, (x, y-25), (x+w, y), (50,50,50), -1)
            cv2.putText(annotated_frame, label, (x+2, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        self.face_eye_closed_times = new_face_eye_closed_times
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), drowsy_any, drowsy_people

    def upload_image(self):
        self.stop_event.set()
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            # Resize image to fit panel if too large
            max_dim = 600
            h, w = img.shape[:2]
            scale = min(max_dim / h, max_dim / w, 1.0)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
            # For static images, just use the old logic (no timer)
            display_frame, drowsy, age, drowsy_people = self.process_frame(img)
            self.show_frame(display_frame)
            if drowsy:
                self.status.config(text='Drowsiness Detected!', fg='red')
            else:
                self.status.config(text='Normal', fg='green')
            self.drowsy_count_label.config(text=f'Drowsy People: {drowsy_people}')

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
        drowsy_people = 0
        annotated_frame = frame.copy()
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            # Age prediction for this face
            age = predict_age(face_img)
            eyes = EYE_CASCADE.detectMultiScale(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY), 1.1, 4)
            eye_states = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_img = face_img[ey:ey+eh, ex:ex+ew]
                state = predict_eye_state(eye_img)
                eye_states.append(state)
                color = (0,255,0) if state else (0,0,255)
                cv2.rectangle(annotated_frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), color, 2)
            if len(eye_states) == 2 and sum(eye_states) == 0:
                drowsy_people += 1
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0,0,255), 2)
                label = f'Drowsy | Age: {age}'
            else:
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0,255,0), 2)
                label = f'Normal | Age: {age}'
            # Draw label above face box
            cv2.rectangle(annotated_frame, (x, y-25), (x+w, y), (50,50,50), -1)
            cv2.putText(annotated_frame, label, (x+2, y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # If no faces, fallback to old logic
        if len(faces) == 0:
            eyes = EYE_CASCADE.detectMultiScale(gray, 1.1, 4)
            eye_states = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_img = frame[ey:ey+eh, ex:ex+ew]
                state = predict_eye_state(eye_img)
                eye_states.append(state)
                color = (0,255,0) if state else (0,0,255)
                cv2.rectangle(annotated_frame, (ex, ey), (ex+ew, ey+eh), color, 2)
            drowsy = (len(eye_states) == 2 and sum(eye_states) == 0)
        else:
            drowsy = (drowsy_people > 0)
        # For GUI labels, just show first face's age if any
        gui_age = ''
        if len(faces) > 0:
            gui_age = predict_age(frame[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]])
        return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), drowsy, gui_age, drowsy_people

    def show_frame(self, frame):
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)

    def on_close(self):
        self.stop_event.set()
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = DrowsinessApp(root)
    root.protocol('WM_DELETE_WINDOW', app.on_close)
    root.mainloop()
