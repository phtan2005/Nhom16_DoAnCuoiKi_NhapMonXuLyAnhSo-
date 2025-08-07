import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from ultralytics import YOLO
import os

# Load model YOLO
model = YOLO("runs/detect/train/weights/best.pt")

def detect_image():
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    image = cv2.imread(file_path)
    results = model.predict(source=image, imgsz=640, conf=0.5)
    person_count = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        if class_id == 0:
            person_count += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(image, f"Persons: {person_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Detected Persons (Image)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_video():
    file_path = filedialog.askopenfilename(
        title="Chọn video", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not file_path:
        return

    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không mở được video")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_name = "output_detected.mp4"
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=640, conf=0.5, verbose=False)
        person_count = 0

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if class_id == 0:
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"Persons: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)
        cv2.imshow("YOLO Detection (Video)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Hoàn thành", f"Video đã lưu tại: {os.path.abspath(output_name)}")


# Tạo giao diện
root = tk.Tk()
root.title("Đếm người với YOLOv11")
root.geometry("350x220")
root.configure(bg="#f0f0f0")  # Màu nền

label = tk.Label(root, text="Chọn chức năng", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
label.pack(pady=20)

btn_img = tk.Button(root, text="🖼️ Đếm người trong Ảnh", command=detect_image, width=30, height=2, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
btn_img.pack(pady=10)

btn_vid = tk.Button(root, text="🎥 Đếm người trong Video", command=detect_video, width=30, height=2, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
btn_vid.pack(pady=10)


root.mainloop()
