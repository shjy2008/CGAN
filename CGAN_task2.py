import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import time
import pandas as pd
import os
from IPython import display

# ------- Please modify this ----------
is_training = False # Set to True if want to train
generate_letter = 'C' # Valid value: ABCDEFGHIKLMNOPQRSTUVWXY (No J and Z)
#--------------------------------------

MODEL_SAVE_PATH = "myModel_task2.keras"

NUM_CLASSES = 24 # only 24 letters, except J and Z

all_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" # 26 letters
letters = "ABCDEFGHIKLMNOPQRSTUVWXY" # only 24 letters, except J and Z
vowels = "AEIOU"

# index: (0-23) in letters (except J and Z)
def index_to_letter(index):
    return letters[index]

# index: (0-23) in letters (except J and Z)
def letter_to_index(letter):
    return letters.index(letter)

# label: the label in the csv table, 0-24 but doesn't have 9
def label_to_index(label):
    return letter_to_index(all_letters[label])

# label: the label in the csv table, 0-24 but doesn't have 9
def is_label_a_vowel(label):
    return all_letters[label] in vowels

# Load the CSV file
train_data_csv = pd.read_csv('sign_mnist_train.csv')
y_train = train_data_csv.iloc[:, 0].values
x_train = train_data_csv.iloc[:, 1:].values
x_train = x_train.reshape(-1, 28, 28)
x_train = x_train.astype('float32') / 255.0 
x_train = x_train[..., tf.newaxis]

test_data_csv = pd.read_csv('sign_mnist_test.csv')
y_test = test_data_csv.iloc[:, 0].values
x_test = test_data_csv.iloc[:, 1:].values
x_test = x_test.reshape(-1, 28, 28)
x_test = x_test.astype('float32') / 255.0
x_test = x_test[..., tf.newaxis]

# Convert label to index
y_train = np.array([label_to_index(label) for label in y_train])
y_test = np.array([label_to_index(label) for label in y_test])

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = True)

@tf.keras.utils.register_keras_serializable()
class CGAN(tf.keras.Model):
    """Conditional Generative Adversarial Network"""

    def __init__(self, num_classes, latent_dim = 100, **kwargs):
        super(CGAN, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        
        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape = (self.latent_dim + self.num_classes, )),
                tf.keras.layers.Dense(7*7*256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Reshape((7, 7, 256)),
                tf.keras.layers.Conv2DTranspose(128, 5, strides = 1, padding = 'same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(64, 5, strides = 2, padding = 'same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2DTranspose(1, 5, strides = 2, padding = 'same', activation = 'sigmoid')     
            ]
        )

        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape = (28, 28, 1 + self.num_classes)),
                tf.keras.layers.Conv2D(64, 5, strides = 2, padding = 'same', input_shape = [28, 28, 1]),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(128, 5, strides = 2, padding = 'same'),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ]
        )

        @tf.function
        def sample(self, eps = None):
            if eps == None:
                eps = tf.random.normal(shape = (100, self.latent_dim))
            return self.generator(eps, training = False)
        
    
class CGAN_trainer():
    def __init__(self, train_images, train_labels, num_classes, latent_dim = 100, batch_size = 256):
        self.cgan = CGAN(num_classes, latent_dim)
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.batch_size = batch_size
        self.dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(train_images.shape[0]).batch(self.batch_size)
        self.num_batches = train_images.shape[0] // self.batch_size
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        return fake_loss
    
    @tf.function
    def train_step(self, images, labels):
        noise = tf.random.normal([images.shape[0], self.latent_dim])
        labels_one_hot = tf.one_hot(labels, self.num_classes)

        generator_input = tf.concat([noise, labels_one_hot], axis = 1)

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.cgan.generator(generator_input, training = True)
                        
            # Reshape labels to match image dimensions
            labels_channel = tf.reshape(labels_one_hot, [-1, 1, 1, self.num_classes])  # [batch_size, 1, 1, num_classes]
            labels_channel = tf.tile(labels_channel, [1, 28, 28, 1])  # Tile to [batch_size, 28, 28, num_classes]

            # Concatenate images and labels along the channel dimension
            real_images_with_labels = tf.concat([images, labels_channel], axis=-1)
            fake_images_with_labels = tf.concat([generated_images, labels_channel], axis=-1)

            real_output = self.cgan.discriminator(real_images_with_labels, training = True)
            fake_output = self.cgan.discriminator(fake_images_with_labels, training = True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.cgan.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.cgan.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.cgan.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.cgan.discriminator.trainable_variables))
        return gen_loss, disc_loss

    def train(self, epochs):
        if os.path.exists(MODEL_SAVE_PATH):
            self.load_model()
            print ("Load previous model success, continue to train based on the previous model")

        self.gen_losses = []
        self.disc_losses = []
        for epoch in range(epochs):
            start_time = time.time()

            gen_loss = 0
            disc_loss = 0
            for image_batch, label_batch in self.dataset:
                gl, dl = self.train_step(image_batch, label_batch)
                gen_loss += gl
                disc_loss += dl
            
            gen_loss /= self.num_batches
            disc_loss /= self.num_batches

            self.gen_losses.append(gen_loss)
            self.disc_losses.append(disc_loss)

            print(f'Epoch {epoch + 1}, Gen Loss: {gen_loss}, Disc Loss: {disc_loss}, Time: {time.time() - start_time}')

            # Produce images for the GIF as we go
            # display.clear_output(wait = True)
            # print('Time for epoch {} is {} desc'.format(epoch + 1, time.time() - start_time))

            self.generate_and_save_images(epoch)

            self.save_model()
        
        # Generate after the final epoch
        # display.clear_output(wait = True)
        # self.generate_and_save_images(epochs)
    
    def generate_and_save_images(self, epoch):
        num_examples = 24
        noise = tf.random.normal([num_examples, self.latent_dim])
        labels = tf.constant(list(range(num_examples)))
        labels_one_hot = tf.one_hot(labels, self.num_classes)
        generator_input = tf.concat([noise, labels_one_hot], axis=1)
        predictions = self.cgan.generator(generator_input, training = False)

        fig = plt.figure(figsize = (6, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(6, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap = 'gray')
            plt.title(index_to_letter(i))
            plt.axis("off")
        
        plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
        # plt.show()

        # fig, ax1 = plt.subplots(1, 1, sharey = True, figsize = (10, 5))
        # ax1.plot(self.gen_losses, label = 'Generator Loss')
        # ax1.plot(self.disc_losses, label = 'Discriminator Loss')
        # ax1.legend()
        # plt.show()

    def save_model(self):
        self.cgan.save(MODEL_SAVE_PATH)
    
    def load_model(self):
        self.cgan = tf.keras.models.load_model(MODEL_SAVE_PATH)


cgan = CGAN_trainer(x_train, y_train, NUM_CLASSES)
cgan.train(epochs = 50)



# print(y_train[0], y_test[0])
