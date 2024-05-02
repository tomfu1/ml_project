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
    targets = []
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
            loss, latent_mean, latent_log_var, reconstructed_output, latent_vector, padded_hamiltonian_data, gradient = result

            batch_total_loss += loss
            if batch_gradients is None:
                batch_gradients = gradient
            else:
                for k, v in gradient.items():
                    batch_gradients[k] += v

            if batch_count % config.batch_size == 0 or sample_idx == num_samples - 1:
                batch_total_loss /= batch_count
                total_loss += batch_total_loss
                for k, v in batch_gradients.items():
                    batch_gradients[k] /= batch_count 
                cvae.update_parameters(batch_gradients, config.learning_rate)
                logging.debug(f'''Batch {epoch + 1} - {batch_no}:
  Loss: {batch_total_loss}
  Duration: {datetime.now() - batch_start}''')
                batch_total_loss = 0  # Reset batch loss for the next batch
                batch_gradients = None  # Reset batch gradients for the next batch
                batch_no += 1
                batch_count = 0
                batch_start = datetime.now()

            batch_count += 1

        logging.info(f"Epoch {epoch + 1} Average Loss: {total_loss / batch_no}")

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

class nablaVAE:
    def __init__(self, hamiltonian_input_shape, config):
        # Initialize network parameters based on input shape and latent dimension
        self.latent_dim = config.latent_dim

        # Encoder parameters
        self.encoder_fc1_weights = torch.randn(np.prod(hamiltonian_input_shape), 64, device=device)
        self.encoder_fc2_weights_mean = torch.randn(64, self.latent_dim, device=device)
        self.encoder_fc2_weights_log_var = torch.randn(64, self.latent_dim, device=device)
        self.encoder_fc1_bias = torch.zeros(64, device=device)
        self.encoder_fc2_bias_mean = torch.zeros(self.latent_dim, device=device)
        self.encoder_fc2_bias_log_var = torch.zeros(self.latent_dim, device=device)

        # Decoder parameters
        self.decoder_fc1_weights = torch.randn(self.latent_dim, 64, device=device)
        self.decoder_fc1_bias = torch.zeros(64, device=device)

        # Convolutional layers and kernels for the encoder
        self.conv1_weights = torch.randn(*config.encoder_convolution1_kernel, device=device)
        self.conv2_weights = torch.randn(*config.encoder_convolution2_kernel, device=device)
        self.conv1_bias = torch.zeros(hamiltonian_input_shape, device=device)
        self.conv2_bias = torch.zeros(hamiltonian_input_shape, device=device)

        # Convolutional layers and kernels for the decoder
        self.conv3_weights = torch.randn(*config.decoder_convolution1_kernel, device=device)
        self.conv4_weights = torch.randn(*config.decoder_convolution2_kernel, device=device)
        self.conv3_bias = torch.zeros(hamiltonian_input_shape, device=device)
        self.conv4_bias = torch.zeros(hamiltonian_input_shape, device=device)

        # Output layers for the decoder
        self.decoder_fc2_weights = torch.randn(64, 128, device=device)
        self.decoder_fc3_weights = torch.randn(128, np.prod(hamiltonian_input_shape), device=device)
        self.decoder_fc2_bias = torch.zeros(128, device=device)
        self.decoder_fc3_bias = torch.zeros(np.prod(hamiltonian_input_shape), device=device)

        # Initialize optimizer
        self.optimizer = AdamOptimizer(config)

        # Save input shape for later use
        self.inputShape = hamiltonian_input_shape

        self.parameters = {
            'encoder_fc1_weights': torch.randn(np.prod(hamiltonian_input_shape), 64, device=device),
            'encoder_fc2_weights_mean': torch.randn(64, self.latent_dim, device=device),
            'encoder_fc2_weights_log_var': torch.randn(64, self.latent_dim, device=device),
            'encoder_fc1_bias': torch.zeros(64, device=device),
            'encoder_fc2_bias_mean': torch.zeros(self.latent_dim, device=device),
            'encoder_fc2_bias_log_var': torch.zeros(self.latent_dim, device=device),
            'decoder_fc1_weights': torch.randn(self.latent_dim, 64, device=device),
            'decoder_fc1_bias': torch.zeros(64, device=device),
            'decoder_fc2_weights': torch.randn(64, 128, device=device),
            'decoder_fc3_weights': torch.randn(128, np.prod(hamiltonian_input_shape), device=device),
            'decoder_fc2_bias': torch.zeros(128, device=device),
            'decoder_fc3_bias': torch.zeros(np.prod(hamiltonian_input_shape), device=device),
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
        energy_data = 0

        ## Total loss
        total_loss = reconstruction_loss + kl_divergence + energy_data

        # Backward pass and parameter update
        gradients = self.backward(
            padded_hamiltonian_data,
            latent_vector,
            latent_mean,
            latent_log_var,
            energy_data,
            config,
        )

        return (
            total_loss,
            latent_mean,
            latent_log_var,
            reconstructed_output,
            latent_vector,
            padded_hamiltonian_data,
            gradients,
        )

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
            k: torch.zeros_like(getattr(self, k), device=device)
            for k in [
                'encoder_fc1_weights',
                'encoder_fc1_bias',
                'encoder_fc2_weights_mean',
                'encoder_fc2_bias_mean',
                'encoder_fc2_weights_log_var',
                'encoder_fc2_bias_log_var',
                'decoder_fc1_weights',
                'decoder_fc1_bias',
                'decoder_fc2_weights',
                'decoder_fc2_bias',
                'decoder_fc3_weights',
                'decoder_fc3_bias',
            ]
        }

        reconstruction_loss_gradient = -(hamiltonian_data - self.decoder_forward(latent_vector, config))
        reconstruction_loss_gradient = reconstruction_loss_gradient.flatten()
        decoder_fc3_activation_gradient = reconstruction_loss_gradient

        # Compute gradient with respect to weights of decoder_fc3
        parameter_gradients['decoder_fc3_weights'] = (
            decoder_fc3_activation_gradient * self.decoder_fc3_weights
        ).sum(-1)

        # Compute gradient with respect to bias of decoder_fc3
        parameter_gradients['decoder_fc3_bias'] = torch.sum(decoder_fc3_activation_gradient, axis=0)

        reshaped_outputfc3 = self.pad_output_for_backward_pass(
            decoder_fc3_activation_gradient,
            self.decoder_fc2_weights.shape,
        )

        # Compute gradient of reconstruction loss with respect to activations of decoder_fc2
        decoder_fc2_activation_gradient = torch.matmul(reshaped_outputfc3, self.decoder_fc2_weights.T)
        decoder_fc2_activation_gradient *= self.leaky_relu_derivative(
            decoder_fc2_activation_gradient,
            config.leaky_relu_derivative_alpha,
        )

        # Compute gradient with respect to weights of decoder_fc2
        parameter_gradients['decoder_fc2_weights'] = torch.matmul(
            decoder_fc2_activation_gradient,
            self.decoder_fc2_weights,
        )

        # Compute gradient with respect to bias of decoder_fc2
        parameter_gradients['decoder_fc2_bias'] = torch.sum(decoder_fc2_activation_gradient, axis=0)

        # Compute gradient of reconstruction loss with respect to activations of decoder_fc1
        decoder_fc1_activation_gradient = torch.matmul(decoder_fc2_activation_gradient, self.decoder_fc1_weights.T)
        decoder_fc1_activation_gradient *= self.leaky_relu_derivative(
            decoder_fc1_activation_gradient,
            config.leaky_relu_derivative_alpha,
        )

        # Compute gradient with respect to weights of decoder_fc1
        parameter_gradients['decoder_fc1_weights'] = torch.matmul(decoder_fc1_activation_gradient, self.decoder_fc1_weights)

        # Compute gradient with respect to bias of decoder_fc1
        parameter_gradients['decoder_fc1_bias'] = torch.sum(decoder_fc1_activation_gradient, axis=0)

        # Compute gradient of KL divergence term
        kl_div_mean_gradient = 0.5 * (2 * latent_mean)
        kl_div_log_var_gradient = 0.5 * (1 - torch.exp(latent_log_var))

        num_elements = len(reconstruction_loss_gradient)
        padding_size = len(reconstruction_loss_gradient) - len(kl_div_mean_gradient)

        padded_kl_div_mean_gradient = F.pad(
            kl_div_mean_gradient,
            (0, padding_size),
            mode='constant',
        )
        padded_kl_div_log_var_gradient = F.pad(
            kl_div_log_var_gradient,
            (0, padding_size),
            mode='constant',
        )
 
        encoder_fc2_mean_gradient = reconstruction_loss_gradient * padded_kl_div_mean_gradient
        encoder_fc2_log_var_gradient = reconstruction_loss_gradient * padded_kl_div_log_var_gradient

        target_size = (100 * 64) * ((encoder_fc2_mean_gradient.shape[0] + (100 * 64) - 1) // (100 * 64))

        # Pad the gradients to match the target size
        padding_size = target_size - len(encoder_fc2_mean_gradient)
        padded_gradients_mean = F.pad(encoder_fc2_mean_gradient, (0, padding_size), mode='constant')
        padded_gradients_log_var = F.pad(encoder_fc2_log_var_gradient, (0, padding_size), mode='constant')

        # Reshape the padded gradients to match the shape of the FC2 encoder layer
        reshaped_gradients_mean = padded_gradients_mean[:100*64].reshape((100, 64))
        reshaped_gradients_log_var = padded_gradients_log_var[:100*64].reshape((100, 64))

        # Compute gradient with respect to weights of encoder_fc2_mean
        parameter_gradients['encoder_fc2_weights_mean'] = torch.matmul(self.encoder_fc2_weights_mean, reshaped_gradients_mean)

        # Compute gradient with respect to bias of encoder_fc2_mean
        parameter_gradients['encoder_fc2_bias_mean'] = torch.sum(reshaped_gradients_mean, axis=0)

        # Compute gradient with respect to weights of encoder_fc2_log_var
        parameter_gradients['encoder_fc2_weights_log_var'] = torch.matmul(
            self.encoder_fc2_weights_log_var,
            reshaped_gradients_log_var,
        )

        ## Compute gradient with respect to bias of encoder_fc2_log_var
        parameter_gradients['encoder_fc2_bias_log_var'] = torch.sum(reshaped_gradients_log_var, axis=0)

        # Compute gradient of encoder_fc1
        encoder_fc1_activation_gradient = torch.matmul(
            reshaped_gradients_mean,
            self.encoder_fc1_weights.T,
        )
        encoder_fc1_activation_gradient += torch.matmul(
            reshaped_gradients_log_var, 
            self.encoder_fc1_weights.T,
        )
        encoder_fc1_activation_gradient *= self.leaky_relu_derivative(
            encoder_fc1_activation_gradient,
            config.leaky_relu_derivative_alpha,
        )

        # Compute gradient with respect to weights of encoder_fc1
        parameter_gradients['encoder_fc1_weights'] = torch.matmul(
            self.encoder_fc1_weights.T,
            encoder_fc1_activation_gradient.T,
        )

        # Compute gradient with respect to bias of encoder_fc1
        parameter_gradients['encoder_fc1_bias'] = torch.sum(encoder_fc1_activation_gradient, axis=0)

        return parameter_gradients

    def convolve2d(self, input_data, kernel, bias):
        input_height, input_width = input_data.shape
        kernel_height, kernel_width = kernel.shape

        # Calculate output dimensions
        output_height = input_height - kernel_height + 1
        output_width = input_width - kernel_width + 1

        # Initialize output matrix
        output = torch.zeros((output_height, output_width), device=device)

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
        #conv1_output = self.convolve2d(hamiltonian_data, self.conv1_weights, self.conv1_bias)
        #conv1_output = self.leaky_relu(conv1_output, config.leaky_relu_alpha)

        #conv2_output = self.convolve2d(conv1_output, self.conv2_weights, self.conv2_bias)
        #conv2_output = self.leaky_relu(conv2_output, config.leaky_relu_alpha)

        ## Flatten the output of the convolutional layers
        flattened_output = hamiltonian_data.flatten()

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

    def leaky_relu_derivative(self, x, alpha):
        return torch.where(x > 0, 1, alpha)

    def normalize_data(self, data):
        return data / torch.max(torch.abs(data))

    def min_max_normalize(self, x):
        min_val = torch.min(x)
        max_val = torch.max(x)
        return (x - min_val) / (max_val - min_val)

    def pad_output_for_backward_pass(self, output, target_shape):
        num_elements = output.shape[0]
        closest_multiple = int(np.ceil(num_elements / 8192) * 8192)
        # Calculate the padding size
        padding_size = closest_multiple - num_elements
        target_shape = (closest_multiple // target_shape[1], target_shape[1])

        # Pad the output tensor with zeros if necessary
        if padding_size > 0:
            padded_output = F.pad(output, (0, padding_size), mode='constant')
        else:
            padded_output = output
        reshaped_output = padded_output.reshape(target_shape)

        return reshaped_output

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn(*mu.shape, device=device)
        return mu + torch.exp(0.5 * log_var) * epsilon

    def update_parameters(self, gradients, learning_rate):
        zipped = zip(gradients.items(), self.parameters.items())
        for (param_name, param_gradient), (param_name, param_value) in zipped:
            # Update parameters based on gradients
            self.parameters[param_name] -= learning_rate * param_value


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
        leaky_relu_derivative_alpha=0.01,
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
        self.leaky_relu_derivative_alpha = leaky_relu_derivative_alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.start_row_idx = start_row_idx
        self.test_size = test_size
        self.use_gpu = use_gpu

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='main.yaml')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(stream=sys.stderr)
    # For some reason have to do this line separately for logging.DEBUG to play nice)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        with open(args.config) as f:
            config = yaml.load(f, yaml.Loader)
    except FileNotFoundError:
        config = {}
    except Exception as e:
        logging.warning(e)
        config = {}
    config = Config(**(config or {}))

    if config.use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    logging.info(f'device: {device}')

    main(config)
