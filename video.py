# author: Panas Pokharel
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Import necessary libraries
from predict import imagePredict  # Import image prediction function
import cv2  # OpenCV library for computer vision
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx


# Function to detect faces in a video stream
def detect_faces(printRes):
    # Load the pre-trained Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open video capture using the default camera (index 0)
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read a frame from the video capture
        ret, frame = video_capture.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame using the Haar cascade classifier
        # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
        # Higher values increase the chance of detecting faces but may also result in false positives.
        # minNeighbors: Parameter specifying how many neighbors a region should have to be considered a face.
        # Higher values decrease the chance of false positives but may miss some faces.
        # minSize: Minimum possible object size. Objects smaller than this size are ignored.
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the region of interest (last detected face)
            last_face = frame[y:y+h, x:x+w]

            # Save the last detected face as an image in a specified directory
            cv2.imwrite('./fa/far.jpg', last_face)

            # Draw a rectangle around the detected face on the original frame
            # Parameters: image, top-left corner coordinates, bottom-right corner coordinates, color (in BGR format),
            # thickness of the rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Call the image prediction function to predict something based on the last detected face
            prediction = imagePredict()
            print(prediction)
            printRes(prediction)


        # Display the original frame with the detected faces
        cv2.imshow('Video', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Release the video capture object
    video_capture.release()

    # Close all windows
    cv2.destroyAllWindows()

    return prediction

# Call the detect_faces function to start face detection
# detect_faces()


# author: Panas Pokharel