# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx necessary imports xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

def training_model(epoch_count=20, batch_size=32):
    # storing Directory containing the training images in a variable
    image_dir = './images/'
    # Number of classes
    class_count = 6
    # Data augmentation and preprocessing for training images
    data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    # Generate batches of augmented training data
    training_generator = data_generator.flow_from_directory(
        image_dir,
        target_size=(224, 224),  # Reshape images to this size
        batch_size=batch_size,
        class_mode='categorical'  # Use one-hot encoded labels
    )
    # Create a sequential model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),  # Convolutional layer 1
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 1
        layers.Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer 2
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 2
        layers.Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer 3
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 3
        layers.Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer 4
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 4
        layers.Conv2D(64, (3, 3), activation='relu'),  # Convolutional layer 5
        layers.MaxPooling2D((2, 2)),  # Max pooling layer 5
        layers.Flatten(),  # Flatten the output for dense layers
        layers.Dropout(0.5),  # Apply dropout regularization
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),  # Dense layer with regularization
        layers.Dense(class_count, activation='softmax')  # Output layer with softmax activation
    ])
    # Compile the model with optimizerRMSprop and metrics as accuracy
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])
    # Train the model and fit the model to the training data
    model.fit(training_generator, epochs=epoch_count)  
    # Save the trained model insode model folder and override of already exists
    model.save('./model/')
    # Return the trained model
    return model

# calls function
# training_model() 