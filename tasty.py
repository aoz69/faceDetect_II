import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def train_classification_model(num_epochs=10, batch_size=32):
    train_images = './images/'
    train_labels = './images/'
    num_classes = 3
    # Preprocess the data and define the model architecture
    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_images,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_generator, epochs=num_epochs)
    model.save('./')
    return model


# Example usage


train_classification_model()
