import cv2
from deepface import DeepFace
import os
import tkinter as tk
from tkinter import filedialog

class FaceVerifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Face Verifier')

        # UI Components
        self.reference_img_path = ''
        self.compare_img_path = ''

        self.label_reference = tk.Label(master, text="Select Reference Image:")
        self.label_reference.pack()
        self.btn_reference = tk.Button(master, text="Browse", command=self.select_reference_image)
        self.btn_reference.pack()

        self.label_compare = tk.Label(master, text="Select Compare Image:")
        self.label_compare.pack()
        self.btn_compare = tk.Button(master, text="Browse", command=self.select_compare_image)
        self.btn_compare.pack()

        self.btn_verify = tk.Button(master, text="Verify Faces", command=self.verify_faces)
        self.btn_verify.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def select_reference_image(self):
        self.reference_img_path = filedialog.askopenfilename()
        self.label_reference.config(text=f"Reference Image: {self.reference_img_path}")

    def select_compare_image(self):
        self.compare_img_path = filedialog.askopenfilename()
        self.label_compare.config(text=f"Compare Image: {self.compare_img_path}")

    def verify_faces(self):
        if self.reference_img_path and self.compare_img_path:
            result = DeepFace.verify(img1_path=self.reference_img_path,
                                     img2_path=self.compare_img_path,
                                     detector_backend='retinaface',
                                     model_name='ArcFace')

            if result['verified']:
                self.result_label.config(text=f"Face Match Found! (Distance: {result['distance']:.2f})")
            else:
                self.result_label.config(text=f"Face Did Not Match. (Distance: {result['distance']:.2f})")
        else:
            self.result_label.config(text="Please select both images.")

# 메인 실행
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceVerifierApp(root)
    root.mainloop()
