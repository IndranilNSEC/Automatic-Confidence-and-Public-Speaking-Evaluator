import cv2
from fer import FER
from collections import deque, Counter

def process_video(video):
    # Set up video capture
    detector = FER(mtcnn=True)
    cap = cv2.VideoCapture(video)

    # History of emotions
    emotion_window = deque(maxlen=50)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions
        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]["emotions"]
            dominant = max(emotions, key=emotions.get)
            emotion_window.append(dominant)

            # Draw emotion label
            cv2.putText(frame, f'Emotion: {dominant}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show video
        cv2.imshow("Emotion Detector", frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Wrap up
    cap.release()
    cv2.destroyAllWindows()

    # Summary
    summary = Counter(emotion_window)
    dominant_emotion = summary.most_common(1)[0][0]
    print(f"\nDominant Emotion in Session: {dominant_emotion}")
    
    return dominant_emotion

#process_video("Videos/VID-20240911-WA0007.mp4")