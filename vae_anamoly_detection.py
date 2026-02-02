"""
===========================================================
Project: Variational Autoencoder (VAE) for Anomaly Detection
Author: Sivaneswari R
===========================================================

DESCRIPTION:
This project implements a Variational Autoencoder (VAE) to detect
anomalies in the MNIST dataset. The model is trained only on normal
digits and tested on corrupted/anomalous samples. Reconstruction
error is used to identify anomalies.

Technologies:
Python, TensorFlow/Keras, NumPy, Matplotlib, Scikit-learn
===========================================================
"""

# ==============================
# IMPORT LIBRARIES
# ==============================
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# ==============================
# LOAD DATASET
# ==============================
(x_train, _), (x_test, _) = mnist.load_data()

# normalize
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))


# ==============================
# CREATE SYNTHETIC ANOMALIES
# ==============================
def create_anomalies(data, fraction=0.2):
    noisy = data.copy()
    n = int(len(data) * fraction)

    idx = np.random.choice(len(data), n)
    noisy[idx] = np.random.random(noisy[idx].shape)

    labels = np.zeros(len(data))
    labels[idx] = 1 # anomaly label

    return noisy, labels


x_test_noisy, y_labels = create_anomalies(x_test)


# ==============================
# VAE MODEL
# ==============================

latent_dim = 2

# Encoder
inputs = layers.Input(shape=(784,))
h = layers.Dense(256, activation='relu')(inputs)

z_mean = layers.Dense(latent_dim)(h)
z_log_var = layers.Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args
    eps = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * eps


z = layers.Lambda(sampling)([z_mean, z_log_var])


# Decoder
decoder_h = layers.Dense(256, activation='relu')
decoder_out = layers.Dense(784, activation='sigmoid')

h_decoded = decoder_h(z)
outputs = decoder_out(h_decoded)

vae = tf.keras.Model(inputs, outputs)


# ==============================
# VAE LOSS FUNCTION
# ==============================
recon_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
recon_loss = tf.reduce_mean(recon_loss) * 784

kl_loss = -0.5 * tf.reduce_mean(
    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
)

vae_loss = recon_loss + kl_loss
vae.add_loss(vae_loss)

vae.compile(optimizer='adam')


# ==============================
# TRAIN
# ==============================
print("Training VAE...")
vae.fit(x_train, epochs=10, batch_size=128, validation_split=0.1)


# ==============================
# RECONSTRUCTION ERROR
# ==============================
reconstructed = vae.predict(x_test_noisy)

errors = np.mean(np.square(x_test_noisy - reconstructed), axis=1)


# ==============================
# ANOMALY DETECTION
# ==============================
threshold = np.percentile(errors, 80)

preds = (errors > threshold).astype(int)

auc = roc_auc_score(y_labels, errors)

print("AUC Score:", auc)


# ==============================
# VISUALIZATION
# ==============================
plt.hist(errors, bins=50)
plt.title("Reconstruction Error Distribution")
plt.show()


print("Project Completed Successfully!")