import tensorflow as tf
from tensorflow import keras


def predict_image(model, image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1)
    confidence = tf.reduce_max(predictions).numpy()

    return predicted_class, confidence


# Example usage
image_path = './test/688.png'

# Load the trained model
model = keras.models.load_model('./')

# Make prediction
predicted_class, confidence = predict_image(model, image_path)

# Print the predicted class and confidence
print(f"Predicted Class: {predicted_class}")
print(f"Confidence: {confidence}")
