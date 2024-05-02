#!/usr/bin/env python

from datetime import datetime
import logging
import sys

from nablaDFT.dataset import hamiltonian_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import yaml

device = "cpu"

def main(config):
    train = hamiltonian_dataset.HamiltonianDatabase(config.dataset_path)
    padded_X_train, padded_X_test, y_train, y_test, maxSize = process_dataset(train, config)

    logging.info(f'Max hamiltonian shape: {maxSize["H"]}')

    cvae = nablaVAE(maxSize["H"], config)
    train_model(cvae, padded_X_train, y_train, config)

def process_dataset(train_dataset, config, random_state=42):
    logging.info(f'Processing dataset ({config.start_row_idx} -> {config.end_row_idx}) ...')
    max_hamiltonian_shape = None
    features = []
    targets=[]
    excluded_samples = 0  # Counter for excluded samples

    max_size = {"H":(0,0)}

    for sample_idx in range(config.start_row_idx, config.end_row_idx):
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
            logging.warning(f"Error processing sample {sample_idx}: {e}. This sample will be excluded.")
            excluded_samples += 1

    logging.info(f"Excluded {excluded_samples} samples due to dimension mismatch errors.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=config.test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, max_size

def train_model(cvae, X_train, y, config):
    num_samples = len(X_train)

    for epoch in range(config.num_epochs):
        logging.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        total_loss = 0
        batch_total_loss = 0
        batch_gradients = None
        batch_count = 1
        batch_no = 1
        batch_start = datetime.now()

        for sample_idx in range(0, num_samples):
            hamiltonian_data = X_train[sample_idx]
            energy = y[sample_idx]

            result = cvae.train_step(hamiltonian_data, energy, config)

            #loss, latent_mean, latent_log_var, reconstructed_output, latent_vector, padded_hamiltonian_data = cvae.train_step(hamiltonian_data, energy)
            #batch_total_loss += loss
            #print(f"input: {hamiltonian_data}")
            #print(reconstructed_output)
            #current_gradients = cvae.backward(padded_hamiltonian_data, latent_vector, latent_mean, latent_log_var, energy)

            ## Accumulate gradients for the batch
            #if batch_gradients is None:
            #    batch_gradients = current_gradients
            #else:
            #    for key in batch_gradients:
            #        batch_gradients[key] += current_gradients[key]

            ## Update parameters after processing each batch
            if batch_count % config.batch_size == 0 or sample_idx == num_samples - 1:
            #    cvae.update_parameters(batch_gradients)
                batch_total_loss /= batch_count
                logging.info(f'Batch {batch_no}:\n  Loss: {batch_total_loss}\n  Duration: {datetime.now() - batch_start}')
                batch_total_loss = 0  # Reset batch loss for the next batch
                batch_gradients = None  # Reset batch gradients for the next batch
                batch_no += 1
                batch_count = 0
                batch_start = datetime.now()

            batch_count += 1

        logging.info(f"Epoch {epoch + 1} Total Loss: {total_loss}")

class AdamOptimizer:
    def __init__(self, config):
        self.learning_rate = config.adam_learning_rate
        self.beta1 = config.adam_beta1
        self.beta2 = config.adam_beta2
        self.epsilon = config.adam_epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, gradients):
        if self.m is None:
            self.m = {
                param_name: np.zeros_like(param_value)
                for param_name, param_value in gradients.items()
            }
            self.v = {
                param_name: np.zeros_like(param_value)
                for param_name, param_value in gradients.items()
            }

        self.t += 1
        for param_name, gradient in gradients.items():
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradient
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            gradients[param_name] = update
        return gradients

class Config:
    def __init__(
        self,
        *,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        adam_learning_rate=0.001,
        batch_size=32,
        dataset_path='dataset_train_2k.db',
        decoder_convolution1_kernel=[3, 3],
        decoder_convolution2_kernel=[3, 3],
        encoder_convolution1_kernel=[3, 3],
        encoder_convolution2_kernel=[3, 3],
        end_row_idx=1000,
        latent_dim=100,
        leaky_relu_alpha=0.2,
        learning_rate=0.0001,
        num_epochs=25,
        start_row_idx=0,
        test_size=0.2,
        use_gpu=True,
    ):
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.adam_learning_rate = adam_learning_rate
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.decoder_convolution1_kernel = decoder_convolution1_kernel
        self.decoder_convolution2_kernel = decoder_convolution2_kernel
        self.encoder_convolution1_kernel = encoder_convolution1_kernel
        self.encoder_convolution2_kernel = encoder_convolution2_kernel
        self.end_row_idx = end_row_idx
        self.latent_dim = latent_dim
        self.leaky_relu_alpha = leaky_relu_alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.start_row_idx = start_row_idx
        self.test_size = test_size
        self.use_gpu = use_gpu

class nablaVAE:
    def __init__(self, hamiltonian_input_shape, config):
        # Initialize network parameters based on input shape and latent dimension
        self.latent_dim = config.latent_dim

        # Encoder parameters
        self.encoder_fc1_weights = to_torch(np.random.randn(np.prod(hamiltonian_input_shape), 64))
        self.encoder_fc2_weights_mean = to_torch(np.random.randn(64, self.latent_dim))
        self.encoder_fc2_weights_log_var = to_torch(np.random.randn(64, self.latent_dim))
        self.encoder_fc1_bias = to_torch(np.zeros(64))
        self.encoder_fc2_bias_mean = to_torch(np.zeros(self.latent_dim))
        self.encoder_fc2_bias_log_var = to_torch(np.zeros(self.latent_dim))

        # Decoder parameters
        self.decoder_fc1_weights = to_torch(np.random.randn(self.latent_dim, 64))
        self.decoder_fc1_bias = to_torch(np.zeros(64))

        # Convolutional layers and kernels for the encoder
        self.conv1_weights = to_torch(np.random.randn(*config.encoder_convolution1_kernel))
        self.conv2_weights = to_torch(np.random.randn(*config.encoder_convolution2_kernel))
        self.conv1_bias = to_torch(np.zeros(hamiltonian_input_shape))
        self.conv2_bias = to_torch(np.zeros(hamiltonian_input_shape))

        # Convolutional layers and kernels for the decoder
        self.conv3_weights = to_torch(np.random.randn(*config.decoder_convolution1_kernel))
        self.conv4_weights = to_torch(np.random.randn(*config.decoder_convolution2_kernel))
        self.conv3_bias = to_torch(np.zeros(hamiltonian_input_shape))
        self.conv4_bias = to_torch(np.zeros(hamiltonian_input_shape))

        # Output layers for the decoder
        self.decoder_fc2_weights = to_torch(np.random.randn(64, 128))
        self.decoder_fc3_weights = to_torch(np.random.randn(128, np.prod(hamiltonian_input_shape)))
        self.decoder_fc2_bias = to_torch(np.zeros(128))
        self.decoder_fc3_bias = to_torch(np.zeros(np.prod(hamiltonian_input_shape)))

        # Initialize optimizer
        self.optimizer = AdamOptimizer(config)

        # Save input shape for later use
        self.inputShape = hamiltonian_input_shape

        self.parameters = {
            'encoder_fc1_weights': to_torch(np.random.randn(np.prod(hamiltonian_input_shape), 64)),
            'encoder_fc2_weights_mean': to_torch(np.random.randn(64, self.latent_dim)),
            'encoder_fc2_weights_log_var': to_torch(np.random.randn(64, self.latent_dim)),
            'encoder_fc1_bias': to_torch(np.zeros(64)),
            'encoder_fc2_bias_mean': to_torch(np.zeros(self.latent_dim)),
            'encoder_fc2_bias_log_var': to_torch(np.zeros(self.latent_dim)),
            'decoder_fc1_weights': to_torch(np.random.randn(self.latent_dim, 64)),
            'decoder_fc1_bias': to_torch(np.zeros(64)),
            'conv1_weights': to_torch(np.random.randn(*config.encoder_convolution1_kernel)),
            'conv2_weights': to_torch(np.random.randn(*config.encoder_convolution2_kernel)),
            'conv1_bias': to_torch(np.zeros(hamiltonian_input_shape)),
            'conv2_bias': to_torch(np.zeros(hamiltonian_input_shape)),
            'conv3_weights': to_torch(np.random.randn(*config.decoder_convolution1_kernel)),
            'conv4_weights': to_torch(np.random.randn(*config.decoder_convolution2_kernel)),
            'conv3_bias': to_torch(np.zeros(hamiltonian_input_shape)),
            'conv4_bias': to_torch(np.zeros(hamiltonian_input_shape)),
            'decoder_fc2_weights': to_torch(np.random.randn(64, 128)),
            'decoder_fc3_weights': to_torch(np.random.randn(128, np.prod(hamiltonian_input_shape))),
            'decoder_fc2_bias': to_torch(np.zeros(128)),
            'decoder_fc3_bias': to_torch(np.zeros(np.prod(hamiltonian_input_shape))),
        }

    def train_step(self, hamiltonian_data, energy_data, config):
        # Forward pass--
        hamiltonian_data = torch.from_numpy(hamiltonian_data).to(device)
        hamiltonian_data = self.min_max_normalize(hamiltonian_data)
        latent_mean, latent_log_var = self.encoder_forward(hamiltonian_data, config)
        latent_vector = self.reparameterize(latent_mean, latent_log_var)
        reconstructed_output = self.decoder_forward(latent_vector, config)

        padded_hamiltonian_data = F.pad(hamiltonian_data, (
            0, reconstructed_output.shape[0] - hamiltonian_data.shape[0],
            0, reconstructed_output.shape[1] - hamiltonian_data.shape[1],
        ))


        ## Compute reconstruction loss
        reconstruction_loss = torch.mean((padded_hamiltonian_data - reconstructed_output) ** 2)

        ## Compute KL divergence
        kl_divergence = -0.5 * torch.mean(1 + latent_log_var - latent_mean ** 2 - torch.exp(latent_log_var))

        ## Energy regularization term

        ##nergy_data= self.normalize_data(energy_data)
        energy_data= 0

        ## Total loss
        total_loss = reconstruction_loss + kl_divergence + energy_data

        ## Backward pass and parameter update
        gradients = self.backward(
            padded_hamiltonian_data,
            latent_vector,
            latent_mean,
            latent_log_var,
            energy_data,
            config,
        )
        #self.update_parameters(gradients)

        #return total_loss, latent_mean, latent_log_var, reconstructed_output, latent_vector, padded_hamiltonian_data

    def backward(
        self,
        hamiltonian_data,
        latent_vector,
        latent_mean,
        latent_log_var,
        energy_data,
        config,
    ):
        # Initialize gradients
        parameter_gradients = {
            'encoder_fc1_weights': to_torch(np.zeros_like(self.encoder_fc1_weights)),
            'encoder_fc1_bias': to_torch(np.zeros_like(self.encoder_fc1_bias)),
            'encoder_fc2_weights_mean': to_torch(np.zeros_like(self.encoder_fc2_weights_mean)),
            'encoder_fc2_bias_mean': to_torch(np.zeros_like(self.encoder_fc2_bias_mean)),
            'encoder_fc2_weights_log_var': to_torch(np.zeros_like(self.encoder_fc2_weights_log_var)),
            'encoder_fc2_bias_log_var': to_torch(np.zeros_like(self.encoder_fc2_bias_log_var)),
            'decoder_fc1_weights': to_torch(np.zeros_like(self.decoder_fc1_weights)),
            'decoder_fc1_bias': to_torch(np.zeros_like(self.decoder_fc1_bias)),
            'conv1_weights': to_torch(np.zeros_like(self.conv1_weights)),
            'conv1_bias': to_torch(np.zeros_like(self.conv1_bias)),
            'conv2_weights': to_torch(np.zeros_like(self.conv2_weights)),
            'conv2_bias': to_torch(np.zeros_like(self.conv2_bias)),
            'conv3_weights': to_torch(np.zeros_like(self.conv3_weights)),
            'conv3_bias': to_torch(np.zeros_like(self.conv3_bias)),
            'conv4_weights': to_torch(np.zeros_like(self.conv4_weights)),
            'conv4_bias': to_torch(np.zeros_like(self.conv4_bias)),
            'decoder_fc2_weights': to_torch(np.zeros_like(self.decoder_fc2_weights)),
            'decoder_fc2_bias': to_torch(np.zeros_like(self.decoder_fc2_bias)),
            'decoder_fc3_weights': to_torch(np.zeros_like(self.decoder_fc3_weights)),
            'decoder_fc3_bias': to_torch(np.zeros_like(self.decoder_fc3_bias)),
        }

        reconstruction_loss_gradient = -(hamiltonian_data - self.decoder_forward(latent_vector, config))
        
        # Note: if a, b are np.arrays, and c, d are equivalent torch.Tensors, and b.ndim >= 2, then
        # np.dot(a, b) =~ (c * d.T).sum(-1)
        # torch.dot(a, b) only supports 1d tensors
        parameter_gradients['decoder_fc3_weights'] = (
            reconstruction_loss_gradient.T * latent_vector.T
        ).sum(-1)
        #parameter_gradients['decoder_fc3_weights'] = np.dot(
        #    reconstruction_loss_gradient.T,
        #    latent_vector,
        #)
        #parameter_gradients['decoder_fc3_bias'] = torch.sum(reconstruction_loss_gradient, axis=0)

        #reconstruction_loss_gradient = reconstruction_loss_gradient.flatten()

        ## Backpropagate gradients through fully connected layers
        #decoder_fc2_gradient = np.dot(reconstruction_loss_gradient, self.decoder_fc3_weights.T)
        #parameter_gradients['decoder_fc2_weights'] = np.dot(latent_vector.T, decoder_fc2_gradient)
        #parameter_gradients['decoder_fc2_bias'] = np.sum(decoder_fc2_gradient, axis=0)

        #decoder_fc1_gradient = np.dot(decoder_fc2_gradient, self.decoder_fc1_weights.T)
        #parameter_gradients['decoder_fc1_weights'] = np.dot(latent_vector.T, decoder_fc1_gradient)
        #parameter_gradients['decoder_fc1_bias'] = np.sum(decoder_fc1_gradient, axis=0)

        #encoder_fc2_log_var_gradient = -0.5 * (1 + latent_log_var - latent_mean ** 2 - np.exp(latent_log_var))
        #parameter_gradients['encoder_fc2_weights_log_var'] = np.dot(latent_vector.T, encoder_fc2_log_var_gradient)
        #parameter_gradients['encoder_fc2_bias_log_var'] = np.sum(encoder_fc2_log_var_gradient, axis=0)

        #encoder_fc2_mean_gradient = np.dot(latent_vector.T, latent_mean)
        #parameter_gradients['encoder_fc2_weights_mean'] = encoder_fc2_mean_gradient
        #parameter_gradients['encoder_fc2_bias_mean'] = np.sum(latent_mean, axis=0)

        #encoder_fc1_gradient = np.dot(latent_mean, self.encoder_fc2_weights_mean.T) +
        #                   np.dot(encoder_fc2_log_var_gradient, self.encoder_fc2_weights_log_var.T)
        #parameter_gradients['encoder_fc1_weights'] = np.dot(hamiltonian_data.T, encoder_fc1_gradient)
        #parameter_gradients['encoder_fc1_bias'] = np.sum(encoder_fc1_gradient, axis=0)

        ## Backpropagate gradients through convolutional layers
        #conv3_weights_gradient, conv3_bias_gradient = self.convolve2d_backprop(hamiltonian_data, reconstruction_loss_gradient, self.conv3_weights)
        #parameter_gradients['conv3_weights'] = conv3_weights_gradient
        #parameter_gradients['conv3_bias'] = conv3_bias_gradient

        #conv4_weights_gradient, conv4_bias_gradient = self.convolve2d_backprop(hamiltonian_data, reconstruction_loss_gradient, self.conv4_weights)
        #parameter_gradients['conv4_weights'] = conv4_weights_gradient
        #parameter_gradients['conv4_bias'] = conv4_bias_gradient

        ## Compute gradients for output layer


        #return parameter_gradients

    def convolve2d(self, input_data, kernel, bias):
        input_height, input_width = input_data.shape
        kernel_height, kernel_width = kernel.shape

        # Calculate output dimensions
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        # Initialize output matrix
        output = to_torch(np.zeros((output_height, output_width)))

        # Perform convolution with bias addition
        for i in range(kernel_height):
            for j in range(kernel_width):
                data_times_weight = kernel[i, j] * input_data
                output += data_times_weight[i:i + output_height, j:j + output_width]

        output += bias[:output_height, :output_width]

        return output

    def decoder_forward(self, latent_vector, config):
        # Forward pass through the decoder
        fc1_output = (latent_vector * self.decoder_fc1_weights.T).sum(-1) + self.decoder_fc1_bias
        fc1_output = self.leaky_relu(fc1_output, config.leaky_relu_alpha)

        fc1_output = self.min_max_normalize(fc1_output)

        fc2Output = (fc1_output * self.decoder_fc2_weights.T).sum(-1) + self.decoder_fc2_bias
        leakyFc2Output = self.leaky_relu(fc2Output, config.leaky_relu_alpha)

        #fc2_outputFinal = self.min_max_normalize(leakyFc2Output)
        final_output = (leakyFc2Output * self.decoder_fc3_weights.T).sum(-1) + self.decoder_fc3_bias
        output_hamiltonian = final_output.reshape(self.inputShape)

        return output_hamiltonian
    
    def encoder_forward(self, hamiltonian_data, config):
        # Apply the convolutional layers
        conv1_output = self.convolve2d(hamiltonian_data, self.conv1_weights, self.conv1_bias)
        conv1_output = self.leaky_relu(conv1_output, config.leaky_relu_alpha)

        conv2_output = self.convolve2d(conv1_output, self.conv2_weights, self.conv2_bias)
        conv2_output = self.leaky_relu(conv2_output, config.leaky_relu_alpha)

        ## Flatten the output of the convolutional layers
        flattened_output = conv2_output.flatten()

        padding_size =  np.prod(self.inputShape) - flattened_output.shape[0]
        padded_output = F.pad(flattened_output, (0, padding_size), mode='constant')

        ## Forward pass through the fully connected layers
        ## np.dot(a, b) where b >=2 == (a * b.T).sum(-1)
        fc1_output = (padded_output * self.encoder_fc1_weights.T).sum(-1) + self.encoder_fc1_bias
        fc1_output = self.leaky_relu(fc1_output, config.leaky_relu_alpha)

        latent_mean = (fc1_output * self.encoder_fc2_weights_mean.T).sum(-1) + self.encoder_fc2_bias_mean
        latent_mean = self.leaky_relu(latent_mean, config.leaky_relu_alpha)
        latent_mean = self.min_max_normalize(latent_mean)

        latent_log_var = (fc1_output * self.encoder_fc2_weights_log_var.T).sum(-1) + self.encoder_fc2_bias_log_var
        latent_log_var = self.leaky_relu(latent_log_var, config.leaky_relu_alpha)
        latent_log_var = self.min_max_normalize(latent_log_var)

        return latent_mean, latent_log_var

    def leaky_relu(self, x, alpha):
        return torch.where(x > 0, x, alpha * x)

    def min_max_normalize(self, x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        return (x - min_val) / (max_val - min_val)

    def reparameterize(self, mu, log_var):
        epsilon = to_torch(np.random.randn(*mu.shape))
        return mu + torch.exp(0.5 * log_var) * epsilon

def to_torch(x):
    if x.dtype == 'float64':
        x = x.astype('float32')
    return torch.from_numpy(x).to(device)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    try:
        with open('main.yaml') as f:
            config = yaml.load(f, yaml.Loader)
    except FileNotFoundError:
        config = {}
    except Exception as e:
        logging.warning(e)
        config = {}
    config = Config(**config)

    if config.use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    logging.info(f'device: {device}')

    main(config)
