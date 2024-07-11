import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3  # TTS library

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' '}

# Buffer to store predicted characters and sentences
predicted_sentence = ""
previous_character = ""
last_character_time = time.time()
character_stability_time = 1.0  # Time in seconds to consider a character stable
last_stable_character = ""
last_stable_time = time.time()

# Initialize TTS engine
engine = pyttsx3.init()

# OpenCV window settings
WINDOW_NAME = 'Hand Gesture Typing'
cv2.namedWindow(WINDOW_NAME)
cv2.moveWindow(WINDOW_NAME, 100, 100)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to detect the hand (you can adjust the range based on your lighting conditions)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(frame_hsv, lower_skin, upper_skin)

    # Apply the mask to the frame
    hand_segment = cv2.bitwise_and(frame, frame, mask=mask)
    background = np.full((H, W, 3), 255, dtype=np.uint8)  # White background
    masked_hand = cv2.addWeighted(background, 1, hand_segment, 1, 0)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                masked_hand,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        current_character = labels_dict[int(prediction[0])]

        if current_character == last_stable_character:
            if time.time() - last_stable_time >= character_stability_time:
                if time.time() - last_character_time >= character_stability_time:
                    predicted_sentence += current_character
                    previous_character = current_character
                    last_character_time = time.time()
        else:
            last_stable_character = current_character
            last_stable_time = time.time()

        cv2.rectangle(masked_hand, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(masked_hand, current_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Display the predicted sentence on the screen
    # Adjust font size dynamically based on the length of predicted_sentence
    font_size = 1.0
    thickness = 2
    text_size = cv2.getTextSize(f'Sentence: {predicted_sentence}', cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]
    
    while text_size[0] > (W - 20):
        font_size -= 0.1
        if font_size < 0.5:  # Ensure font size doesn't go below a minimum threshold
            font_size = 0.5
            break
        thickness -= 1
        if thickness < 1:  # Ensure thickness doesn't go below a minimum threshold
            thickness = 1
            break
        text_size = cv2.getTextSize(f'Sentence: {predicted_sentence}', cv2.FONT_HERSHEY_SIMPLEX, font_size, thickness)[0]

    cv2.putText(masked_hand, f'Sentence: {predicted_sentence}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), thickness, cv2.LINE_AA)

    # Draw buttons for clear and speak options
    cv2.rectangle(masked_hand, (20, 450), (120, 490), (255, 255, 255), -1)
    cv2.rectangle(masked_hand, (140, 450), (240, 490), (255, 255, 255), -1)
    cv2.putText(masked_hand, 'Clear (C)', (30, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(masked_hand, 'Speak (S)', (150, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Concatenate original frame and masked hand image horizontally
    combined_frame = np.hstack((frame, masked_hand))

    cv2.imshow(WINDOW_NAME, combined_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):  # Clear the sentence
        predicted_sentence = ""
    elif key == ord('s'):  # Speak the sentence
        engine.say(predicted_sentence)
        engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
