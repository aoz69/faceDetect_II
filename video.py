
import cv2

def detect_faces():
    # Load the pre-trained face detection cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Start capturing video from the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        # Read each frame of the video
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame with detected faces
        cv2.imshow('Video', frame)
        # Save the image of the last detected face
        i=0
        if len(faces) > 0:
            last_face = frame[y:y+h, x:x+w]
            # cv2.imwrite('./detected/tempImage.jpg', last_face)

            image_path = './images/model/{}.jpg'.format(i)
            cv2.imwrite(image_path, last_face)
            i += 1

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    video_capture.release()
    cv2.destroyAllWindows()


detect_faces()