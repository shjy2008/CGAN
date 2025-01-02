import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from IPython import display
from CGAN_model import CGAN, CGAN_trainer, ModelData

# ------- Please modify this ----------
is_training = True # Set to True if want to train
generate_vowel = False # True(or 1): Vowel, False(or 0): Consonant
#--------------------------------------

MODEL_SAVE_PATH = "myModel_task1.keras"

NUM_CLASSES = 2 # vowel or consonant

x_train, y_train, x_test, y_test = ModelData.load_csv_file()

# Convert label to 0(consonant) or 1(vowel)
y_train = np.array([1 if ModelData.is_label_a_vowel(label) else 0 for label in y_train])
y_test = np.array([1 if ModelData.is_label_a_vowel(label) else 0 for label in y_test])

# Given 0 or 1, generate consonants(0) or vowel(1) images
def generate_image_vowel_or_consonant(is_vowel):
    # myModel = tf.keras.models.load_model(ModelData.MODEL_SAVE_PATH)
    myModel = CGAN(num_classes=NUM_CLASSES, latent_dim=ModelData.LATENT_DIM)
    myModel.load_weights(MODEL_SAVE_PATH)
    y = 1 if is_vowel else 0

    num_examples = 24
    generated_images = myModel.sample([y] * num_examples)

    title = "Vowel" if is_vowel else "Consonant"
    
    fig = plt.figure(figsize=(13, 10))
    for i in range(num_examples):
        plt.subplot(5, 6, i + 1)
        plt.imshow(generated_images[i][:, :, 0], cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.savefig(f'generated_{title}.png')
    # plt.close()
    plt.show()

# If is training, create a trainer to train the model
if is_training:
    cgan = CGAN_trainer(x_train, y_train, NUM_CLASSES, ModelData.LATENT_DIM, model_save_path = MODEL_SAVE_PATH)
    cgan.train(epochs = 50)
else: # If not, only generate images with existing model
    generate_image_vowel_or_consonant(generate_vowel)




