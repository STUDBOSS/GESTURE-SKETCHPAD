**Gesture Sketchpad**
is an interactive drawing application that allows users to sketch on a virtual canvas using hand gestures detected through a webcam. The application utilizes OpenCV for video capture and image processing, MediaPipe for real-time hand tracking, and a machine learning model for shape prediction. Users can seamlessly switch between freehand drawing, shape detection, and an eraser tool by simply using specific hand gestures and keyboard shortcuts.

Features
Hand Gesture Drawing: Use hand movements to draw on the canvas. The application tracks the position of your index finger and thumb to control the drawing.
Shape Detection Mode: Toggle shape detection with the press of the 'B' key. The application predicts the shapes you draw (e.g., circles, squares) and replaces them with accurately predicted shapes.
Eraser Functionality: Erase drawn elements by activating the eraser tool with the 'E' key. The eraser works by covering unwanted parts of the drawing with a white circle, without disturbing shape detection.
Multiple Color Options: Draw with various colors of brushes and pencils to make your sketches more dynamic.
Shape Drawing Activation: Shape detection is triggered when the index and middle fingers are raised, while normal doodling occurs with the thumb and index finger.
Intuitive Controls: Toggle shape detection ('B'), start/stop detection ('D'/'S'), and clear the canvas ('C') easily using the keyboard.
