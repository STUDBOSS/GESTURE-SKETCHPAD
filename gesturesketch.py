import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math
import pickle
import tkinter as tk
from tkinter import Button
import threading

# Load the saved shape prediction model
with open('best_shape_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Function to convert the image to grayscale and apply blurring
def gray_and_blurring(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.blur(gray_img, (15, 15))  # Apply blurring
    return gray_img

# Function to binarize the image
def binarize(img):
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    return img_bin

# Function to find the contours of the largest objects in the binary image
def contours(img_bin, num_shapes=2):
    contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_shapes]
        return sorted_contours
    return None

# Function to extract the histogram of the contour directions
def histogram(img_bin, contour):
    hist = np.zeros((8,))
    if contour is None:
        return hist

    code = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }

    for i in range(len(contour) - 1):
        x1, y1 = contour[i][0]
        x2, y2 = contour[i + 1][0]
        dx = x2 - x1
        dy = y2 - y1
        if (dx, dy) in code:
            hist[code[(dx, dy)]] += 1

    return hist / hist.sum()

# Function to predict the shape of an image using the loaded model
def prediction(img, contour, model):
    mapping = {0: 'square', 1: 'circle', 2: 'triangle'}
    hist = histogram(img, contour)
    predicted_class = model.predict([hist])[0]
    return mapping[predicted_class]

# Function to clear rough shapes before drawing the predicted shape
def clear_rough_shape(paintWindow, bounding_box):
    top_left, bottom_right = bounding_box
    paintWindow[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = 255  # Setting it to white

# Function to draw predicted shape at the same position
def draw_predicted_shape_at_position(predicted_shape, whiteboard, bounding_box):
    top_left, bottom_right = bounding_box
    if predicted_shape == 'square':
        cv2.rectangle(whiteboard, top_left, bottom_right, (0, 255, 0), 7)
    elif predicted_shape == 'circle':
        center = ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
        radius = min(bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]) // 2
        cv2.circle(whiteboard, center, radius, (0, 255, 0), 7)
    elif predicted_shape == 'triangle':
        points = np.array([[top_left[0] + (bottom_right[0] - top_left[0]) // 2, top_left[1]],
                           [top_left[0], bottom_right[1]],
                           [bottom_right[0], bottom_right[1]]], np.int32)
        cv2.polylines(whiteboard, [points], isClosed=True, color=(0, 255, 0), thickness=7)

# Initialize arrays to handle points for drawing
points = [deque(maxlen=1024)]

# Set default color for drawing (blue in this case)
color = (255, 0, 0)  # Blue

# Canvas setup (whiteboard)
paintWindow = np.ones((471, 636, 3), dtype=np.uint8) * 255  # White canvas for drawing

# Initialize MediaPipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.9, min_tracking_confidence=0.9)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True

# Flag to control hand detection and eraser mode
detection_active = True
eraser_active = False

# GUI setup using tkinter
def setup_gui():
    gui_window = tk.Tk()
    gui_window.title("Drawing Controls")
    gui_window.geometry("200x535")

    toggle_detection_icon = tk.PhotoImage(file='toggle.png')  # Make sure the path is correct
    clear_canvas_icon = tk.PhotoImage(file='clear.png')
    toggle_eraser_icon = tk.PhotoImage(file='eraser.png')
    color_blue_icon = tk.PhotoImage(file='blue.png')
    color_green_icon = tk.PhotoImage(file='green.png')
    color_red_icon = tk.PhotoImage(file='red.png')
    shape_mode_icon = tk.PhotoImage(file='shapemode.png')

    # Button actions
    def toggle_detection():
        global detection_active
        detection_active = not detection_active

    def clear_canvas():
        global points, paintWindow
        points = [deque(maxlen=1024)]
        paintWindow[:] = 255  # Set the entire canvas to white

    def set_color(color_choice):
        global color
        color = color_choice

    def toggle_eraser():
        global eraser_active
        eraser_active = not eraser_active  # Toggle eraser mode

    def shape_mode():
        global shape_detection_active
        shape_detection_active = not shape_detection_active
    
    
    def predict_shapes():
        global paintWindow
        gray_img = gray_and_blurring(paintWindow)
        bin_img = binarize(gray_img)
        found_contours = contours(bin_img, num_shapes=2)

        if found_contours is not None:
            for cnt in found_contours:
                shape_prediction = prediction(bin_img, cnt, loaded_model)
                print(f"Predicted Shape: {shape_prediction}")

                # Get the bounding box of the drawn shape
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_box = ((x, y), (x + w, y + h))

                points = [deque(maxlen=1024)]
                paintWindow[:] = 255  # Clear the canvas

                # Draw the predicted shape in the same position on the whiteboard
                draw_predicted_shape_at_position(shape_prediction, paintWindow, bounding_box)

    # GUI Buttons
    Button(gui_window, text="Toggle Detection", image=toggle_detection_icon, compound=tk.TOP, command=toggle_detection).pack(pady=10)
    Button(gui_window, text="Clear Canvas", image=clear_canvas_icon, compound=tk.TOP, command=clear_canvas).pack(pady=10)
    Button(gui_window, text="Toggle Eraser", image=toggle_eraser_icon, compound=tk.TOP, command=toggle_eraser).pack(pady=10)
    Button(gui_window, text="Set Color Blue", image=color_blue_icon, compound=tk.TOP, command=lambda: set_color((255, 0, 0))).pack(pady=10)
    Button(gui_window, text="Set Color Green", image=color_green_icon, compound=tk.TOP, command=lambda: set_color((0, 255, 0))).pack(pady=10)
    Button(gui_window, text="Set Color Red", image=color_red_icon, compound=tk.TOP, command=lambda: set_color((0, 0, 255))).pack(pady=10)
    Button(gui_window, text="Shape Mode", image=shape_mode_icon, compound=tk.TOP, command=shape_mode).pack(pady=10)

    gui_window.mainloop()


# Run the GUI in a separate thread
gui_thread = threading.Thread(target=setup_gui)
gui_thread.daemon = True  # This ensures the GUI thread will exit when the main program exits
gui_thread.start()

# Main loop
# Flag to control shape detection
shape_detection_active = False

# Main loop
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Adjust paintWindow to match frame height (480 px)
    paintWindow = cv2.resize(paintWindow, (paintWindow.shape[1], frame.shape[0]))

    if detection_active:
        # Get hand landmark prediction
        result = hands.process(framergb)

        # Process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * 640)
                    lmy = int(lm.y * 480)
                    landmarks.append([lmx, lmy])

                # Draw landmarks on frames (optional)
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            fore_finger = (landmarks[8][0], landmarks[8][1])  # Index finger tip
            thumb = (landmarks[4][0], landmarks[4][1])  # Thumb tip
            cv2.circle(frame, fore_finger, 3, (0, 255, 0), -1)

            # Calculate the distance between thumb and index finger
            distance = math.hypot(fore_finger[0] - thumb[0], fore_finger[1] - thumb[1])

            # Draw or erase based on distance and eraser mode
            if distance < 40:
                if not eraser_active:
                    points[-1].appendleft(fore_finger)  # Draw
                else:
                    cv2.circle(paintWindow, fore_finger, 20, (255, 255, 255), -1)  # Erase
            else:
                if shape_detection_active and len(points[-1]) > 10:  # End of drawing stroke
                    # Predict the shape
                    gray_img = gray_and_blurring(paintWindow)
                    bin_img = binarize(gray_img)
                    found_contours = contours(bin_img, num_shapes=1)

                    if found_contours is not None:
                        for cnt in found_contours:
                            shape_prediction = prediction(bin_img, cnt, loaded_model)

                            # Get the bounding box of the drawn shape
                            x, y, w, h = cv2.boundingRect(cnt)
                            bounding_box = ((x, y), (x + w, y + h))

                            points = [deque(maxlen=1024)]  # Clear points
                            paintWindow[:] = 255  # Clear the canvas

                            # Draw the predicted shape in the same position on the whiteboard
                            draw_predicted_shape_at_position(shape_prediction, paintWindow, bounding_box)
                points.append(deque(maxlen=1024))  # New line on finger move

    # Draw all points on the canvas
    
    for i in range(len(points)):
    # Ensure deque has at least 2 points before drawing lines
        if len(points[i]) < 2:
            continue  # Skip this deque if not enough points

        for j in range(1, len(points[i])):
            if points[i][j - 1] is None or points[i][j] is None:
                continue
            cv2.line(paintWindow, points[i][j - 1], points[i][j], color, 2)


    # Show the frame and canvas
    cv2.imshow("Drawing", paintWindow)
    cv2.imshow("Webcam", frame)

    # Break the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        # Start hand detection
        detection_active = True
    elif key == ord('s'):
        # Stop hand detection
        detection_active = False
    elif key == ord('b'):
        # Toggle shape detection feature
        shape_detection_active = not shape_detection_active
        print(f"Shape detection {'enabled' if shape_detection_active else 'disabled'}")
    elif key == ord('p') and shape_detection_active:
        # Predict the shapes drawn on the whiteboard
        gray_img = gray_and_blurring(paintWindow)
        bin_img = binarize(gray_img)
        found_contours = contours(bin_img, num_shapes=2)

        if found_contours is not None:
            for cnt in found_contours:
                shape_prediction = prediction(bin_img, cnt, loaded_model)
                print(f"Predicted Shape: {shape_prediction}")

                # Get the bounding box of the drawn shape
                x, y, w, h = cv2.boundingRect(cnt)
                bounding_box = ((x, y), (x + w, y + h))

                # Clear the rough shape
                clear_rough_shape(paintWindow, bounding_box)

                # Draw the predicted shape in the same position on the whiteboard
                draw_predicted_shape_at_position(shape_prediction, paintWindow, bounding_box)

# Clean up
cap.release()
cv2.destroyAllWindows()
