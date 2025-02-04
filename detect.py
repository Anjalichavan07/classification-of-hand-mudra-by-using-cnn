import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Suppress TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ImageClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try using DirectShow backend

        self.label = tk.Label(root)
        self.label.pack()

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.capture_button = tk.Button(self.button_frame, text="Capture Image", bg='#F157DC', command=self.capture_image)
        self.capture_button.pack(side=tk.LEFT)

        self.preprocess_button = tk.Button(self.button_frame, text="Preprocess Image",bg='lightgray', command=self.show_preprocessed_image)
        self.preprocess_button.pack(side=tk.LEFT)

        self.segment_button = tk.Button(self.button_frame, text="Segment Image", bg='#F157DC', command=self.show_segmented_image)
        self.segment_button.pack(side=tk.LEFT)

        self.extract_button = tk.Button(self.button_frame, text="Feature Extraction",bg='lightgray', command=self.show_feature_extracted_image)
        self.extract_button.pack(side=tk.LEFT)

        self.classify_button = tk.Button(self.button_frame, text="Classify Image",bg='#F157DC', command=self.classify_image)
        self.classify_button.pack(side=tk.LEFT)

        self.browse_button = tk.Button(self.button_frame, text="Browse Image",bg='lightgray', command=self.browse_image)
        self.browse_button.pack(side=tk.LEFT)

        # Load the trained model
        self.model_path = r'E:\PROCEDURE\handmudras\data_model.h5'
        self.load_trained_model()

        self.capture()

    def capture(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = cv2.resize(self.current_frame, (400, 300))
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.current_frame))
            self.label.config(image=self.photo)
        self.root.after(10, self.capture)

    def capture_image(self):
        if hasattr(self, 'current_frame'):
            self.captured_image = self.current_frame.copy()
            cv2.imwrite("captured_image.jpg", cv2.cvtColor(self.captured_image, cv2.COLOR_RGB2BGR))
            print("Image captured successfully!")
        else:
            messagebox.showerror("Error", "No image captured!")

    def preprocess_image(self, image):
        if image is not None:
            image = cv2.resize(image, (224, 224))  # Resize to (224, 224)
            image = image / 255.0
            return image
        else:
            print("Error: Image is None")
            return None

    def segment_image(self, image):
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, segmented = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            return segmented
        else:
            print("Error: Image is None")
            return None

    def extract_features(self, image):
        if image is not None:
            edges = cv2.Canny(image, 100, 200)
            hist = cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_RGB2BGR)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return edges, hist
        else:
            print("Error: Image is None")
            return None, None

    def show_preprocessed_image(self):
        if hasattr(self, 'captured_image'):
            preprocessed_image = self.preprocess_image(self.captured_image)
            if preprocessed_image is not None:
                preprocessed_image = (preprocessed_image * 255).astype(np.uint8)
                preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_RGBA2GRAY)  # Convert back to BGR for display
                self.display_side_by_side(self.captured_image, preprocessed_image, "Preprocessed Image")
        else:
            messagebox.showerror("Error", "No captured image to preprocess!")

    def show_segmented_image(self):
        if hasattr(self, 'captured_image'):
            segmented_image = self.segment_image(self.captured_image)
            if segmented_image is not None:
                segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for display
                self.display_side_by_side(self.captured_image, segmented_image, "Segmented Image")
        else:
            messagebox.showerror("Error", "No captured image to segment!")

    def show_feature_extracted_image(self):
        if hasattr(self, 'captured_image'):
            edges, hist = self.extract_features(self.captured_image)
            if edges is not None:
                edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR for display
                self.display_side_by_side(self.captured_image, edges, "Feature Extracted Image")
        else:
            messagebox.showerror("Error", "No captured image to extract features!")

    def display_side_by_side(self, original_image, processed_image, title):
        window = Toplevel(self.root)
        window.title(title)

        original_image_resized = cv2.resize(original_image, (300, 300))
        original_image_tk = ImageTk.PhotoImage(image=Image.fromarray(original_image_resized))
        label1 = tk.Label(window, image=original_image_tk)
        label1.image = original_image_tk
        label1.pack(side=tk.LEFT)

        processed_image_resized = cv2.resize(processed_image, (300, 300))
        processed_image_tk = ImageTk.PhotoImage(image=Image.fromarray(processed_image_resized))
        label2 = tk.Label(window, image=processed_image_tk)
        label2.image = processed_image_tk
        label2.pack(side=tk.RIGHT)

        save_button = tk.Button(window, text="Save Image", command=lambda: self.save_image(processed_image, title))
        save_button.pack(side=tk.BOTTOM)

    def save_image(self, image, title):
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")], title=title)
        if file_path:
            cv2.imwrite(file_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"{title} saved at {file_path}")

    def load_trained_model(self):
        self.model = load_model(self.model_path)
        self.class_labels = {0: 'Anjali', 1: 'Ardhachandra',2:'Katskaavardanm',3:'Matsya',4:'Mushti',5:'Pataka',6:'Samputa',7:'Shikara',8:'SHIVALINGA',9:'Vyagraha'}
        print("Model loaded successfully!")

    def preprocess_image_for_model(self, image_array):
        image = cv2.resize(image_array, (224, 224))  # Resize to (224, 224)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        return image

    def classify_image(self):
        if hasattr(self, 'captured_image') and hasattr(self, 'model'):
            preprocessed_image = self.preprocess_image_for_model(self.captured_image)
            prediction = self.model.predict(preprocessed_image)[0]
            self.predicted_class_index = np.argmax(prediction)
            self.predicted_class = self.class_labels[self.predicted_class_index]
            self.confidence = prediction[self.predicted_class_index]
            messagebox.showinfo("Classification Result", f"Captured image classified as: {self.predicted_class} with confidence: {self.confidence:.2f}")
        else:
            messagebox.showerror("Error", "No captured image or trained model found!")

    def browse_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.captured_image = cv2.imread(file_path)
            if self.captured_image is not None:
                self.captured_image = cv2.cvtColor(self.captured_image, cv2.COLOR_BGR2RGB)
                self.display_image(self.captured_image)
            else:
                messagebox.showerror("Error", "Failed to load image!")
        else:
            messagebox.showerror("Error", "No image selected!")

    def display_image(self, image):
        image = cv2.resize(image, (400, 300))
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        self.label.config(image=self.photo)

if __name__ == "__main__":
    root = tk.Tk()
    root.configure(bg="lightgray")
    app = ImageClassifier(root)
    root.mainloop()
