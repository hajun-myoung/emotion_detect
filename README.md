# 😄 Emotion Detection with Face Recognition

A simple yet powerful **emotion detection pipeline** using face detection and deep learning-based facial emotion recognition.

This project detects faces in an image, predicts emotions for each face, and visualizes the results with:

- Bounding boxes
- Top emotion label
- Emotion probability bar charts

---

## 🚀 Features

- 🔍 Face detection using `MTCNN`
- 🧠 Emotion recognition via `EmotiEffLib`
- 📊 Probability-based emotion bar visualization
- 🖼️ Auto-scaling UI (text & bars scale with image size)
- 💾 Result image export

---

## 🧱 Pipeline Overview

```
Input Image
   ↓
Face Detection (MTCNN)
   ↓
Face Cropping
   ↓
Emotion Prediction (EmotiEffLib)
   ↓
Visualization (bbox + label + bar graph)
   ↓
Output Image
```

---

## 📦 Dependencies

Install required packages:

```bash
pip install opencv-python matplotlib numpy facenet-pytorch emotiefflib
```

---

## ⚙️ Usage

### 1. Set input image

```python
SRC_DIR = "./public/images/"
IMAGE_FILENAME = "image2.jpeg"
```

### 2. Run script

```bash
python main.py
```

### 3. Output

- Visualization displayed via matplotlib
- Saved image:

```
./public/images/result.png
```

---

## 🧠 Supported Emotions

Default emotion classes:

```
Anger, Contempt, Disgust, Fear,
Happiness, Neutral, Sadness, Surprise
```

> ⚠️ Note: Emotion classes may vary depending on the model used.  
> The code dynamically adjusts based on model output length.

---

## 🔍 Key Components

### 1. Face Detection

```python
detect_faces(frame, mtcnn)
```

- Uses `MTCNN`
- Filters low-confidence detections
- Applies padding for better cropping
- Ignores too-small faces

---

### 2. Emotion Recognition

```python
fer.predict_emotions(face_img, logits=False)
```

- Returns:
  - `emotions`: sorted labels
  - `scores`: probability distribution

---

### 3. Visualization

```python
draw_label_and_bars(...)
```

Includes:

- Bounding box
- Top emotion label
- Emotion probability bars

📌 UI scales dynamically based on image size:

- Font size
- Bar width
- Row height

---

## 🖼️ Example Output

```
[ Face Bounding Box ]
[ Top Emotion Label ]

Anger      ████░░░░░ 32.14%
Happiness  ███████░░ 67.86%
...
```

---

## 🛠 Configuration

### Detection threshold

```python
prob_threshold = 0.9
```

### Bounding box padding

```python
padding = 10
```

### Minimum face size

```python
min_face_size = 40
```

---

## 📌 Notes

- Runs on CPU by default (`device="cpu"`)
- Supports ONNX-based inference
- Can be extended to:
  - Real-time webcam input
  - Video processing
  - Web app integration (Flask / FastAPI)

---

## 💡 Future Improvements

- 🎥 Real-time webcam inference
- 🌐 API server (FastAPI)
- 📱 Frontend visualization (React)
- 🎯 Better UI/UX for dense face scenes

---

## 🧑‍💻 Author

Denver
