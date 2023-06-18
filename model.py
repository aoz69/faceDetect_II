# author: Panas Pokharel
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import tensorflow as tf
import json as jso
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

def training_model(epoch_count=50, batch_size=32):
    # storing Directory containing the training images in a variable
    image_dir = './images/'
    # Number of classes
    class_count = 6
    # Data augmentation and preprocessing for training images
    data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,  zoom_range =0.2 , horizontal_flip = True, rotation_range = 80 )
    # Generate batches of augmented training data
    training_generator = data_generator.flow_from_directory(
        image_dir,
        target_size=(128,128),  # Reshape images to this size
        batch_size=batch_size,
        class_mode='categorical' # Use one-hot encoded labels
    )
    # Create a sequential model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128,128, 3)),  # Convolutional layer 1
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 1
        layers.Conv2D(64, (4, 4), activation='relu'),  # Convolutional layer 2
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 2
        layers.Conv2D(64, (4, 4), activation='relu'),  # Convolutional layer 3
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 3
        layers.Conv2D(64, (4, 4), activation='relu'),  # Convolutional layer 4
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 4
        layers.Conv2D(64, (4, 4), activation='relu'),  # Convolutional layer 5
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 5
        layers.Flatten(),  # Flatten the output for dense layers
        layers.Dropout(0.5),  # Apply dropout regularization
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # Dense layer with regularization
        layers.Dense(class_count, activation='softmax')  # Output layer with softmax activation
    ])
    # Compile the model with optimizerRMSprop and metrics as accuracy
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    # Train the model and fit the model to the training data
    model.fit(training_generator, epochs=epoch_count)  
    # Save the trained model insode model folder and override of already exists
    model.save('./model/')
    far = {}
    for i,j in zip(training_generator.class_indices.values() ,training_generator.class_indices.keys()):
        far[i] = j 
    # open json file and writes 
    f= open("class.json" ,"w")
    #saves indices as json file
    jso.dump(far, f)
    f.close()
    # Return the trained model
    return model

# calls function
# training_model()

# author: Panas Pokharel
