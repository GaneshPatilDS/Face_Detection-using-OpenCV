import cv2

# Load the Haar Cascade model
haar_cascade = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set the screen width and height
cap.set(3, 640)
cap.set(4, 480)


while True:
    # Read a frame from the camera
    success, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
# cap.release()
# cv2.destroyAllWindows()
