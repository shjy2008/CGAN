import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from IPython import display
from CGAN_model import CGAN, CGAN_trainer, ModelData

# ------- Please modify this ----------
is_training = True # Set to True if want to train
generate_letter = 'C' # Valid value: ABCDEFGHIKLMNOPQRSTUVWXY (No J and Z)
#--------------------------------------

MODEL_SAVE_PATH = "myModel_task2.keras"

NUM_CLASSES = 24 # only 24 letters, except J and Z

x_train, y_train, x_test, y_test = ModelData.load_csv_file()

# Convert label to index
y_train = np.array([ModelData.label_to_index(label) for label in y_train])
y_test = np.array([ModelData.label_to_index(label) for label in y_test])


# Give a specific letter ('A', 'B', 'C', ..), generate images
def generate_image_with_letter(letter):
    letter = letter.upper()
    if letter in ModelData.letters:
        # myModel = tf.keras.models.load_model(MODEL_SAVE_PATH)
        myModel = CGAN(num_classes=NUM_CLASSES, latent_dim=ModelData.LATENT_DIM)
        myModel.load_weights(MODEL_SAVE_PATH)
        index = ModelData.letter_to_index(letter)

        num_examples = 24
        generated_images = myModel.sample([index] * num_examples)
        
        fig = plt.figure(figsize = (13, 10))
        for i in range(num_examples):
            plt.subplot(5, 6, i + 1)
            plt.imshow(generated_images[i][:, :, 0], cmap = 'gray')
            plt.title(f"letter: {letter}")
            plt.axis('off')
        
        plt.savefig(f"generated_{letter}.png")
        plt.show()
    else:
        print(f"Input value invalid, please try again. Valid input: {ModelData.letters}")
    

# If is training, create a trainer to train the model
if is_training:
    cgan = CGAN_trainer(x_train, y_train, NUM_CLASSES, ModelData.LATENT_DIM, model_save_path = MODEL_SAVE_PATH)
    cgan.train(epochs = 50)
else: # If not, only generate images with existing model
    generate_image_with_letter(generate_letter)




