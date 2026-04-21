from typing import List, Tuple
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list


# 모델에 따라 감정 클래스 순서가 다를 수 있으니,
# 실제 출력 길이에 맞춰 잘라 쓰는 방식으로 처리
DEFAULT_EMOTIONS = [
    "Anger", "Contempt", "Disgust", "Fear",
    "Happiness", "Neutral", "Sadness", "Surprise"
]


def detect_faces(
    frame: np.ndarray,
    mtcnn: MTCNN,
    prob_threshold: float = 0.9,
    padding: int = 10,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)

    if bounding_boxes is None or probs is None:
        return []

    h, w, _ = frame.shape
    results = []

    for bbox, prob in zip(bounding_boxes, probs):
        if prob is None or prob < prob_threshold:
            continue

        x1, y1, x2, y2 = bbox.astype(int)

        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if (x2 - x1) < 40 or (y2 - y1) < 40:
            continue

        face_img = frame[y1:y2, x1:x2, :]
        results.append((face_img, (x1, y1, x2, y2)))

    return results


def draw_label_and_bars(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    top_label: str,
    probs: np.ndarray,
    emotion_names: List[str],
):
    x1, y1, x2, y2 = bbox
    h, w, _ = image.shape

    # bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 대표 label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 2

    label_text = top_label
    (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    label_y = max(20, y1 - 8)
    cv2.rectangle(
        image,
        (x1, label_y - th - baseline - 6),
        (x1 + tw + 10, label_y + 4),
        (0, 255, 0),
        -1
    )
    cv2.putText(
        image,
        label_text,
        (x1 + 5, label_y - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA
    )

    # 얼굴 옆 바 그래프 영역
    bar_x = min(x2 + 12, w - 170)
    bar_y = y1
    bar_w = 100
    row_h = 18

    # 너무 아래로 길어지면 위로 조금 당김
    total_h = len(probs) * row_h
    if bar_y + total_h > h:
        bar_y = max(0, h - total_h - 5)

    for i, (name, p) in enumerate(zip(emotion_names, probs)):
        cy = bar_y + i * row_h

        # 감정 이름
        cv2.putText(
            image,
            name,
            (bar_x, cy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # 바 배경
        bg_x = bar_x + 70
        bg_y1 = cy + 2
        bg_y2 = cy + 14
        cv2.rectangle(
            image,
            (bg_x, bg_y1),
            (bg_x + bar_w, bg_y2),
            (80, 80, 80),
            -1
        )

        # 확률 바
        fill_w = int(bar_w * float(p))
        cv2.rectangle(
            image,
            (bg_x, bg_y1),
            (bg_x + fill_w, bg_y2),
            (0, 255, 255),
            -1
        )

        # 퍼센트 텍스트
        score_text = f"{p:.2f}"
        cv2.putText(
            image,
            score_text,
            (bg_x + bar_w + 8, cy + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


SRC_DIR = "./public/images/"
IMAGE_FILENAME = "image.jpeg"

input_file = os.path.join(SRC_DIR, IMAGE_FILENAME)
device = "cpu"
model_name = get_model_list()[0]

frame_bgr = cv2.imread(input_file)
frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

mtcnn = MTCNN(
    keep_all=True,
    post_process=False,
    min_face_size=40,
    device=device,
)

fer = EmotiEffLibRecognizer(
    engine="onnx",
    model_name=model_name,
    device=device,
)

detected_faces = detect_faces(frame, mtcnn)
vis_frame = frame.copy()

for face_img, bbox in detected_faces:
    # logits=False면 softmax 확률 반환
    emotions, scores = fer.predict_emotions(face_img, logits=False)

    top_label = emotions[0]
    probs = np.array(scores[0], dtype=float)

    emotion_names = DEFAULT_EMOTIONS[: len(probs)]

    draw_label_and_bars(
        vis_frame,
        bbox,
        top_label,
        probs,
        emotion_names
    )

plt.figure(figsize=(12, 8))
plt.imshow(vis_frame)
plt.axis("off")
plt.show()
plt.savefig("result.png")
plt.imsave(os.path.join(SRC_DIR, "result.png"), vis_frame)