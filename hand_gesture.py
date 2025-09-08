import cv2
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Helper: check finger states
FINGER_TIPS = [8, 12, 16, 20]
FINGER_PIPS = [6, 10, 14, 18]

def finger_states(lm):
    states = {
        'thumb': lm[4].x < lm[3].x,  # works for one hand (may need flip)
        'index': lm[8].y < lm[6].y,
        'middle': lm[12].y < lm[10].y,
        'ring': lm[16].y < lm[14].y,
        'pinky': lm[20].y < lm[18].y,
    }
    return states

def classify_emotion(states, lm):
    up = [f for f,v in states.items() if v]
    if len(up) == 0:
        return "Angry (Fist)"
    if len(up) == 5:
        return "Calm (Open palm)"
    if states['index'] and states['middle'] and not states['ring'] and not states['pinky']:
        return "Excited (V sign)"
    if states['thumb'] and not any([states['index'], states['middle'], states['ring'], states['pinky']]):
        wrist_y = lm[0].y
        thumb_y = lm[4].y
        return "Happy (Thumbs Up)" if thumb_y < wrist_y else "Sad (Thumbs Down)"
    if states['index'] and not any([states['middle'], states['ring'], states['pinky'], states['thumb']]):
        return "Thinking (Index up)"
    return "Unsure"

# Webcam loop
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    emotion = "No hand"
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            states = finger_states(handLms.landmark)
            emotion = classify_emotion(states, handLms.landmark)

    cv2.putText(frame, f"Emotion: {emotion}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Gesture Emotion", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

