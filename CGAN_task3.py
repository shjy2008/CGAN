import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from IPython import display
from CGAN_model import CGAN, CGAN_trainer, ModelData
import os

# ------- Please modify this ----------
is_training = True # Set to True if want to train
generate_rotation_angle = 90 # Valid values: 0, 90, 180, 270
#--------------------------------------

MODEL_SAVE_PATH = "myModel_task3.keras"

angles = [0, 90, 180, 270]

NUM_CLASSES = 4  # 4 angles

rotated_train_data_path = 'rotated_train_data.npz'
rotated_test_data_path = 'rotated_test_data.npz'

def create_rotated_data(x, y):
    rotated_x = []
    rotated_y = []
    for i in range(len(x)):
        image = x[i]
        rotated_x.append(image)
        rotated_y.append(0)
        for rotate_90_num in range(1, 4): # 1: 90, 2: 180, 3: 270
            rotated_image = tf.image.rot90(image, k = rotate_90_num)
            rotated_x.append(rotated_image)
            rotated_y.append(rotate_90_num)

    rotated_x = np.array(rotated_x)
    rotated_y = np.array(rotated_y)
    return (rotated_x, rotated_y)

if is_training:
    # If the rotated data already exists, only need to load the data
    if os.path.exists(rotated_train_data_path) and os.path.exists(rotated_test_data_path):
        print("Local rotated data exists, do not need to create")
        rotated_train_data = np.load(rotated_train_data_path)
        rotated_x_train = rotated_train_data['x']
        rotated_y_train = rotated_train_data['y']

        rotated_test_data = np.load(rotated_test_data_path)
        rotated_x_test = rotated_test_data['x']
        rotated_y_test = rotated_test_data['y']
    else: # If not exists, create a new data
        print("Creating local rotated data...")

        x_train, y_train, x_test, y_test = ModelData.load_csv_file()

        # Convert label to index
        y_train = np.array([ModelData.label_to_index(label) for label in y_train])
        y_test = np.array([ModelData.label_to_index(label) for label in y_test])

        rotated_x_train, rotated_y_train = create_rotated_data(x_train, y_train)
        rotated_x_test, rotated_y_test = create_rotated_data(x_test, y_test)

        np.savez(rotated_train_data_path, x=rotated_x_train, y=rotated_y_train)
        np.savez(rotated_test_data_path, x=rotated_x_test, y=rotated_y_test)

        print("Finish creating local rotated data")


# Given an angle (0, 90, 180, 270), produce images
def generate_image_with_angle(angle):
    if angle not in angles:
        print (f"Input value invalid, please try again. Valid input: {angles}")
        return
    
    index = angles.index(angle)
    # myModel = tf.keras.models.load_model(MODEL_SAVE_PATH)
    myModel = CGAN(num_classes=NUM_CLASSES, latent_dim=ModelData.LATENT_DIM)
    myModel.load_weights(MODEL_SAVE_PATH)

    num_examples = 24
    generated_images = myModel.sample([index] * num_examples)
    
    fig = plt.figure(figsize=(13, 10))
    for i in range(num_examples):
        plt.subplot(5, 6, i + 1)
        plt.imshow(generated_images[i][:, :, 0], cmap='gray')
        title = f"Angle: {angle}"
        plt.title(title)
        plt.axis('off')

    plt.savefig(f'generated_rotation_{angle}.png')
    plt.show()


# If is training, create a trainer to train the model
if is_training:
    trainer = CGAN_trainer(rotated_x_train, rotated_y_train, NUM_CLASSES, ModelData.LATENT_DIM, model_save_path = MODEL_SAVE_PATH)
    trainer.train(epochs=50)
else: # If not, only generate images with existing model
    generate_image_with_angle(generate_rotation_angle)

