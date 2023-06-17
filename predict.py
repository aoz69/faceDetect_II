# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import tensorflow as tf
from tensorflow import keras
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# image_path = './detected/123.jpg'
def imagePredict():
    # Load the trained model
    model = keras.models.load_model('./model')
    # Path to the test image
    image_path = './test/68.png'
    # Load and preprocess the test image
    # Load the test image using Keras and resize it to the target size of (224, 224)
    test_image = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    # Convert the test image to a NumPy array
    test_image_array = keras.preprocessing.image.img_to_array(test_image)
    # Add a batch dimension to the test image array, as the model expects a batch of images
    test_image_array = tf.expand_dims(test_image_array, 0)  # Add batch dimension
    test_image_array /= 255.0  # Normalize the image
    # Make predictions on the test image
    predictions = model.predict(test_image_array)
    predicted_class = tf.argmax(predictions, axis=1)  # Get the predicted class index
    confidence = tf.reduce_max(predictions).numpy()  # Get the confidence score
    # Print the predicted class and confidence
    print(f"Prediction class is : {predicted_class}")
    print(f"Confidence level of the prediction is : {confidence}")
    # Return the predicted class and confidence
    return predicted_class, confidence

# Call the imagePredict function
imagePredict()