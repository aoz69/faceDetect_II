# author: Panas Pokharel
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import tensorflow as tf
import json as j
from tensorflow import keras
import numpy as num
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# image_path = './detected/123.jpg'
def imagePredict():
    # Load the trained model
    model = keras.models.load_model('./model/')
    # Path to the test image
    image_path = './fa/far.jpg'
    # Load and preprocess the test image
    # Load the test image using Keras and resize it to the target size of (224, 224)
    test_image = keras.utils.load_img(image_path, target_size=(128, 128))
    # Converts test image to a NumPy array
    test_image_array = keras.utils.img_to_array(test_image)
    # Adding a batch dimension
    # model requires a batch of images
    test_image_array = tf.expand_dims(test_image_array, 0)  # Add batch dimension
    test_image_array /= 255.0  # Normalize the image
    #  predictions on the test image provided 
    predictions = model.predict(test_image_array)
    # print(num.argmax(predictions))
    #predicted class index
    predicted_class = tf.argmax(predictions, axis=1)  
    # confidence score
    confidence = tf.reduce_max(predictions).numpy()  
    # Print the predicted class and confidence
    print(f"Prediction class is : {predicted_class}")
    print(f"Confidence level of the prediction is : {confidence}")
    f = open("class.json" , "r")
    load_json = j.load(f)
    # Return the predicted class and confidence
    return  load_json[str(num.argmax(predictions))] , confidence

# Call the imagePredict function
# imagePredict()

# author: Panas Pokharel
