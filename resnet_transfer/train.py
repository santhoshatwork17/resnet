import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import joblib

def train_model(dataset_path, model_output_path="resnet_model.h5"):
    img_size = 224
    batch_size = 32

    # Data Augmentation
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_data = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size),
                                             batch_size=batch_size, class_mode='categorical', subset='training')
    val_data = datagen.flow_from_directory(dataset_path, target_size=(img_size, img_size),
                                           batch_size=batch_size, class_mode='categorical', subset='validation')

    # Load Pretrained ResNet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    
    # Freeze Base Model
    for layer in base_model.layers:
        layer.trainable = False

    # Add Custom Layers
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(len(train_data.class_indices), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train Model
    model.fit(train_data, validation_data=val_data, epochs=5)

    # Save Model and Class Indices
    model.save(model_output_path)
    joblib.dump(train_data.class_indices, "class_indices.pkl")

if __name__ == "__main__":
    train_model("path_to_your_dataset")  # Change this
