#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np

from sklearn.model_selection import train_test_split
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import math

#sys.path.append('/content/nablaDFT')

import nablaDFT

from nablaDFT.dataset import NablaDFT

from nablaDFT.dataset import hamiltonian_dataset

from sklearn.model_selection import train_test_split

def process_dataset(train_dataset, start_idx, end_idx, test_size=0.2, random_state=42):
    max_hamiltonian_shape = None
    features = []
    targets=[]
    excluded_samples = 0  # Counter for excluded samples

    max_size = {"H":(0,0)}


    for sample_idx in range(start_idx, end_idx):
        try:
            sample = train_dataset[sample_idx]
            Z, R, E, _, H, S, _ = sample
            hamiltonian_shape = H.shape

            pad_width_H = max(0, max_size['H'][0] - H.shape[0])
            pad_height_H = max(0, max_size['H'][1] - H.shape[1])
            H_padded = np.pad(H, ((0, pad_width_H), (0, pad_height_H)))

            max_size['H'] = (max(max_size['H'][0], H_padded.shape[0]), max(max_size['H'][1], H_padded.shape[1]))

            features.append(H_padded)
            targets.append(E)
        except ValueError as e:
            print(f"Error processing sample {sample_idx}: {e}. This sample will be excluded.")
            excluded_samples += 1

    print(f"Excluded {excluded_samples} samples due to dimension mismatch errors.")


    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=test_size, random_state=random_state)


    return X_train, X_test, y_train, y_test, max_size


class nablaVAE:
    def __init__(self, hamiltonian_input_shape, latent_dim):
        # Initialize network parameters based on input shape and latent dimension
        self.latent_dim = latent_dim

        # Encoder parameters
        self.encoder_fc1_weights = np.random.randn(np.prod(hamiltonian_input_shape), 64)
        print(f"Initial FC1: {self.encoder_fc1_weights}")
        self.encoder_fc2_weights_mean = np.random.randn(64, latent_dim)
        self.encoder_fc2_weights_log_var = np.random.randn(64, latent_dim)
        self.encoder_fc1_bias = np.zeros(64)
        self.encoder_fc2_bias_mean = np.zeros(latent_dim)
        self.encoder_fc2_bias_log_var = np.zeros(latent_dim)

        # Decoder parameters
        self.decoder_fc1_weights = np.random.randn(latent_dim, 64)
        self.decoder_fc1_bias = np.zeros(64)

        # Convolutional layers and kernels for the encoder
        self.conv1_weights = np.random.randn(3, 3)  # Example kernel size (3x3)
        self.conv2_weights = np.random.randn(3, 3)  # Example kernel size (3x3)
        self.conv1_bias = np.zeros(hamiltonian_input_shape)
        self.conv2_bias = np.zeros(hamiltonian_input_shape)

        # Convolutional layers and kernels for the decoder
        self.conv3_weights = np.random.randn(3, 3)  # Example kernel size (3x3)
        self.conv4_weights = np.random.randn(3, 3)  # Example kernel size (3x3)
        self.conv3_bias = np.zeros(hamiltonian_input_shape)
        self.conv4_bias = np.zeros(hamiltonian_input_shape)

        # Output layers for the decoder
        self.decoder_fc2_weights = np.random.randn(64, 128)
        self.decoder_fc3_weights = np.random.randn(128, np.prod(hamiltonian_input_shape))
        self.decoder_fc2_bias = np.zeros(128)
        self.decoder_fc3_bias = np.zeros(np.prod(hamiltonian_input_shape))

        # Initialize optimizer
        self.optimizer = AdamOptimizer()

        # Save input shape for later use
        self.inputShape = hamiltonian_input_shape

        self.parameters = {
            'encoder_fc1_weights': np.random.randn(np.prod(hamiltonian_input_shape), 64),
            'encoder_fc2_weights_mean': np.random.randn(64, latent_dim),
            'encoder_fc2_weights_log_var': np.random.randn(64, latent_dim),
            'encoder_fc1_bias': np.zeros(64),
            'encoder_fc2_bias_mean': np.zeros(latent_dim),
            'encoder_fc2_bias_log_var': np.zeros(latent_dim),
            'decoder_fc1_weights': np.random.randn(latent_dim, 64),
            'decoder_fc1_bias': np.zeros(64),
            'decoder_fc2_weights': np.random.randn(64, 128),
            'decoder_fc3_weights': np.random.randn(128, np.prod(hamiltonian_input_shape)),
            'decoder_fc2_bias': np.zeros(128),
            'decoder_fc3_bias': np.zeros(np.prod(hamiltonian_input_shape))
        }


    def train_step(self, hamiltonian_data, energy_data, learning_rate=0.0001):

        # Forward pass--

        hamiltonian_data = self.normalize_data(hamiltonian_data)
        latent_mean, latent_log_var = self.encoder_forward(hamiltonian_data)
        latent_vector = self.reparameterize(latent_mean, latent_log_var)
        reconstructed_output = self.decoder_forward(latent_vector)
        latent_mean = self.normalize_data(latent_mean)
        latent_log_var = self.normalize_data(latent_log_var)


        reconstructed_output = self.normalize_data(reconstructed_output)

        padded_hamiltonian_data = np.pad(hamiltonian_data, ((0, reconstructed_output.shape[0] - hamiltonian_data.shape[0]),
                                                    (0, reconstructed_output.shape[1] - hamiltonian_data.shape[1])))
        # Compute reconstruction loss
        reconstruction_loss = np.mean((padded_hamiltonian_data - reconstructed_output) ** 2)

        # Compute KL divergence
        kl_divergence = -0.5 * np.mean(1 + latent_log_var - latent_mean ** 2 - np.exp(latent_log_var))
        energy_data =0

        # Total loss
        total_loss = reconstruction_loss + kl_divergence + energy_data
        gradients = cvae.backward(padded_hamiltonian_data, latent_vector, latent_mean, latent_log_var, energy_data)

        return total_loss, latent_mean, latent_log_var, reconstructed_output, latent_vector, padded_hamiltonian_data, gradients

    def convolve2d(self, input_data, kernel, bias):
        input_height, input_width = input_data.shape
        kernel_height, kernel_width = kernel.shape

        # Calculate output dimensions
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        # Initialize output matrix
        output = np.zeros((output_height, output_width))

       # Perform convolution with bias addition
        for i in range(kernel_height):
            for j in range(kernel_width):
                data_times_weight = kernel[i, j] * input_data
                output += data_times_weight[i:i + output_height, j:j + output_width]

        return output

    def leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    def normalize_data(self, data):
        return data / np.max(np.abs(data))

    def min_max_normalize(self,x):
      min_val = np.min(x)
      max_val = np.max(x)
      return (x - min_val) / (max_val - min_val)


    def encoder_forward(self, hamiltonian_data):
        # Apply convolution layers
        #conv1_output = self.convolve2d(hamiltonian_data, self.conv1_weights, self.conv1_bias)
        #conv1_output = self.leaky_relu(conv1_output)

        #conv2_output = self.convolve2d(conv1_output, self.conv2_weights, self.conv2_bias)
        #conv2_output = self.leaky_relu(conv2_output)

        # Flatten the output of the convolutional layers
        flattened_output = hamiltonian_data.flatten()

        padding_size =  np.prod(self.inputShape) - flattened_output.size
        padded_output = np.pad(flattened_output, (0, padding_size), mode='constant')

        # Forward pass through the fully connected layers
        fc1_output = np.dot(padded_output, self.encoder_fc1_weights) + self.encoder_fc1_bias
        fc1_output = self.leaky_relu(fc1_output)

        latent_mean = np.dot(fc1_output, self.encoder_fc2_weights_mean) + self.encoder_fc2_bias_mean
        latent_mean = self.leaky_relu(latent_mean)
        latent_mean = self.min_max_normalize(latent_mean)

        latent_log_var = np.dot(fc1_output, self.encoder_fc2_weights_log_var) + self.encoder_fc2_bias_log_var
        latent_log_var = self.leaky_relu(latent_log_var)
        latent_log_var = self.min_max_normalize(latent_log_var)

        return latent_mean, latent_log_var

    def decoder_forward(self, latent_vector):
        # Forward pass through the decoder
        fc1_output = np.dot(latent_vector, self.decoder_fc1_weights) + self.decoder_fc1_bias
        fc1_output = self.leaky_relu(fc1_output)

        fc1_output = self.min_max_normalize(fc1_output)

        fc2Output = np.dot(fc1_output, self.decoder_fc2_weights) + self.decoder_fc2_bias
        leakyFc2Output = self.leaky_relu(fc2Output)

        #fc2_outputFinal = self.min_max_normalize(leakyFc2Output)
        final_output = np.dot(leakyFc2Output, self.decoder_fc3_weights) + self.decoder_fc3_bias
        output_hamiltonian = final_output.reshape(self.inputShape)

        return output_hamiltonian

    def backward(self, hamiltonian_data, latent_vector, latent_mean, latent_log_var, energy_data):
    # Initialize gradients
      parameter_gradients = {
        'encoder_fc1_weights': np.zeros_like(self.encoder_fc1_weights),
        'encoder_fc1_bias': np.zeros_like(self.encoder_fc1_bias),
        'encoder_fc2_weights_mean': np.zeros_like(self.encoder_fc2_weights_mean),
        'encoder_fc2_bias_mean': np.zeros_like(self.encoder_fc2_bias_mean),
        'encoder_fc2_weights_log_var': np.zeros_like(self.encoder_fc2_weights_log_var),
        'encoder_fc2_bias_log_var': np.zeros_like(self.encoder_fc2_bias_log_var),
        'decoder_fc1_weights': np.zeros_like(self.decoder_fc1_weights),
        'decoder_fc1_bias': np.zeros_like(self.decoder_fc1_bias),
        'decoder_fc2_weights': np.zeros_like(self.decoder_fc2_weights),
        'decoder_fc2_bias': np.zeros_like(self.decoder_fc2_bias),
        'decoder_fc3_weights': np.zeros_like(self.decoder_fc3_weights),
        'decoder_fc3_bias': np.zeros_like(self.decoder_fc3_bias) }

      reconstruction_loss_gradient = -(hamiltonian_data - self.decoder_forward(latent_vector))
      reconstruction_loss_gradient = reconstruction_loss_gradient.flatten()
      decoder_fc3_activation_gradient = reconstruction_loss_gradient

    # Compute gradient with respect to weights of decoder_fc3
      parameter_gradients['decoder_fc3_weights'] = np.dot(decoder_fc3_activation_gradient, self.decoder_fc3_weights.T)

    # Compute gradient with respect to bias of decoder_fc3
      parameter_gradients['decoder_fc3_bias'] = np.sum(decoder_fc3_activation_gradient, axis=0)

      reshaped_outputfc3 = self.pad_output_for_backward_pass(decoder_fc3_activation_gradient,self.decoder_fc2_weights.shape)

    # Compute gradient of reconstruction loss with respect to activations of decoder_fc2
      decoder_fc2_activation_gradient = np.dot(reshaped_outputfc3, self.decoder_fc2_weights.T)
      decoder_fc2_activation_gradient *= self.leaky_relu_derivative(decoder_fc2_activation_gradient)

    # Compute gradient with respect to weights of decoder_fc2
      parameter_gradients['decoder_fc2_weights'] = np.dot(decoder_fc2_activation_gradient, self.decoder_fc2_weights)

    # Compute gradient with respect to bias of decoder_fc2
      parameter_gradients['decoder_fc2_bias'] = np.sum(decoder_fc2_activation_gradient, axis=0)

    # Compute gradient of reconstruction loss with respect to activations of decoder_fc1
      decoder_fc1_activation_gradient = np.dot(decoder_fc2_activation_gradient, self.decoder_fc1_weights.T)
      decoder_fc1_activation_gradient *= self.leaky_relu_derivative(decoder_fc1_activation_gradient)

    # Compute gradient with respect to weights of decoder_fc1
      parameter_gradients['decoder_fc1_weights'] = np.dot(decoder_fc1_activation_gradient, self.decoder_fc1_weights)

    # Compute gradient with respect to bias of decoder_fc1

      parameter_gradients['decoder_fc1_bias'] = np.sum(decoder_fc1_activation_gradient, axis=0)

# Compute gradient of KL divergence term
      kl_div_mean_gradient = 0.5 * (2 * latent_mean)
      kl_div_log_var_gradient = 0.5 * (1 - np.exp(latent_log_var))

      num_elements = len(reconstruction_loss_gradient)
      padding_size = len(reconstruction_loss_gradient) - len(kl_div_mean_gradient)
      #padded_reconstruction_gradient = np.pad(reconstruction_loss_gradient, (0, padding_size), mode='constant')

      padded_kl_div_mean_gradient = np.pad(kl_div_mean_gradient, (0, padding_size), mode='constant')
      padded_kl_div_log_var_gradient = np.pad(kl_div_log_var_gradient, (0, padding_size), mode='constant')

      encoder_fc2_mean_gradient = reconstruction_loss_gradient * padded_kl_div_mean_gradient
      encoder_fc2_log_var_gradient = reconstruction_loss_gradient * padded_kl_div_log_var_gradient

      target_size = (100 * 64) * ((encoder_fc2_mean_gradient.size + (100 * 64) - 1) // (100 * 64))

      # Pad the gradients to match the target size
      padding_size = target_size - len(encoder_fc2_mean_gradient)
      padded_gradients_mean = np.pad(encoder_fc2_mean_gradient, (0, padding_size), mode='constant')
      padded_gradients_log_var = np.pad(encoder_fc2_log_var_gradient, (0, padding_size), mode='constant')

# Reshape the padded gradients to match the shape of the FC2 encoder layer
      reshaped_gradients_mean = padded_gradients_mean[:100*64].reshape((100, 64))
      reshaped_gradients_log_var = padded_gradients_log_var[:100*64].reshape((100, 64))

# Compute gradient with respect to weights of encoder_fc2_mean
      parameter_gradients['encoder_fc2_weights_mean'] = np.dot(self.encoder_fc2_weights_mean, reshaped_gradients_mean)

# Compute gradient with respect to bias of encoder_fc2_mean
      parameter_gradients['encoder_fc2_bias_mean'] = np.sum(reshaped_gradients_mean, axis=0)

# Compute gradient with respect to weights of encoder_fc2_log_var
      parameter_gradients['encoder_fc2_weights_log_var'] = np.dot(self.encoder_fc2_weights_log_var, reshaped_gradients_log_var)

# Compute gradient with respect to bias of encoder_fc2_log_var
      parameter_gradients['encoder_fc2_bias_log_var'] = np.sum(reshaped_gradients_log_var, axis=0)

# Compute gradient of encoder_fc1
      encoder_fc1_activation_gradient = np.dot(reshaped_gradients_mean, self.encoder_fc1_weights.T) + np.dot(reshaped_gradients_log_var, self.encoder_fc1_weights.T)
      encoder_fc1_activation_gradient *= self.leaky_relu_derivative(encoder_fc1_activation_gradient)

# Compute gradient with respect to weights of encoder_fc1
      parameter_gradients['encoder_fc1_weights'] = np.dot(self.encoder_fc1_weights.T, encoder_fc1_activation_gradient.T)

      # Compute gradient with respect to bias of encoder_fc1
      parameter_gradients['encoder_fc1_bias'] = np.sum(encoder_fc1_activation_gradient, axis=0)

      return parameter_gradients

    def convolve2d_backprop(self, input_data, gradient, kernel):
    # Perform convolution and compute gradients for the weights and bias
      weights_gradient = np.zeros_like(kernel)
      bias_gradient = np.sum(gradient)

      for i in range(input_data.shape[0]):
        for j in range(input_data.shape[1]):
            weights_gradient += input_data[i, j] * gradient

      return weights_gradient, bias_gradient

    def clip_gradients(self, gradients, threshold):
        for key, value in gradients.items():
            gradients[key] = np.clip(value, -threshold, threshold)
        return gradients

    def generate(self, latent_vector):
        reconstructed_output = self.decoder_forward(latent_vector)
        return reconstructed_output

    def update_parameters(self, gradients, learning_rate):
      print("updating parameters")
      for (param_name, param_gradient), (param_name, param_value) in zip(gradients.items(), self.parameters.items()):
        # Update parameters based on gradients
        self.parameters[param_name] -= learning_rate * param_value

    def reparameterize(self, mu, log_var):
        epsilon = np.random.randn(*mu.shape)
        return mu + np.exp(0.5 * log_var) * epsilon

    def pad_output_for_backward_pass(self,output, target_shape):

    # Calculate the closest multiple of 8192 (64*128) greater than or equal to the output size
        num_elements = output.size
        closest_multiple = int(np.ceil(num_elements / 8192) * 8192)

    # Calculate the padding size
        padding_size = closest_multiple - num_elements

        target_shape = (closest_multiple // target_shape[1], target_shape[1])

    # Pad the output tensor with zeros if necessary
        if padding_size > 0:
          padded_output = np.pad(output, (0, padding_size), mode='constant')

        else:
          padded_output = output

        reshaped_output = padded_output.reshape(target_shape)

        return reshaped_output

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    def update(self, gradients):
        if self.m is None:
            self.m = {param_name: np.zeros_like(param_value) for param_name, param_value in gradients.items()}
            self.v = {param_name: np.zeros_like(param_value) for param_name, param_value in gradients.items()}
        self.t += 1
        for param_name, gradient in gradients.items():
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradient
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            gradients[param_name] = update
        return gradients

def train_model(cvae, X_train, y, num_epochs=25, batch_size=32):
    num_samples = len(X_train)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        total_loss = 0
        batch_total_loss = 0
        batch_gradients = None
        num_batches = math.ceil(num_samples / batch_size)

        for sample_idx in range(0, num_samples):
            # Extract the current batch
            batch_hamiltonian_data = X_train[sample_idx]
            batch_energy_data = y[sample_idx]
            # Perform a training step for the current batch
            loss, latent_mean, latent_log_var, reconstructed_output, latent_vector, padded_hamiltonian_data,gradient = cvae.train_step(batch_hamiltonian_data, batch_energy_data)

            batch_total_loss +=loss
            if (sample_idx + 1) % batch_size == 0 or sample_idx == num_samples - 1:
                total_loss += batch_total_loss/batch_size

                if batch_gradients is None:
                    batch_gradients = gradient
                else:
                    for param_name, param_gradient in gradient.items():
                        batch_gradients[param_name] += param_gradient

                if (sample_idx + 1) % batch_size == 0:
                    for param_name, param_gradient in batch_gradients.items():
                        batch_gradients[param_name] /= batch_size

                    cvae.update_parameters(batch_gradients, learning_rate=0.0001)

                    batch_total_loss = 0
                    batch_gradients = None


        epoch_average_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} Average Loss: {epoch_average_loss}")


def calculate_rmsd(coords1, coords2):
    # Calculate the difference between coordinates
    diff = coords1 - coords2

    # Square the differences
    squared_diff = np.square(diff)

    # Calculate the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Take the square root to get RMSD
    rmsd = np.sqrt(mean_squared_diff)

    print(f"RMSD: {rmsd}")

    return rmsd

def psnr(input_matrix, output_matrix):
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((input_matrix - output_matrix) ** 2)

    # Determine the maximum possible pixel value
    max_pixel_value = np.max(input_matrix)

    # Calculate PSNR
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))

    print(f"PSNR: {psnr}")

    return psnr

start_idx = 0
end_idx = 100
train = hamiltonian_dataset.HamiltonianDatabase("dataset_train_2k.db")
padded_X_train, padded_X_test, y_train, y_test, maxSize= process_dataset(train,start_idx,end_idx)
max_hamiltonian_shape = maxSize['H']
print(max_hamiltonian_shape)
latent_dim = 100
cvae = nablaVAE(max_hamiltonian_shape, latent_dim)
train_model(cvae, padded_X_train,y_train)
