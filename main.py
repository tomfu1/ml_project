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
    database = hamiltonian_dataset.HamiltonianDatabase(config.dataset_path)
    X_train, X_test, y_train, y_test = process_dataset(database, config)
    logging.debug(f'Hamiltonian shape: {X_train[0].shape}')

    logging.info(f'Starting training ({len(X_train)} samples) ...')
    train(X_train, y_train, config)

def train(X, y, config):
    cvae = nablaVAE(config, X[0].shape)

    for epoch in range(config.num_epochs):
        epoch_start = datetime.now()
        losses = []
        for batch_no, (X_batch, y_batch) in enumerate(batched(X, y, config.batch_size)):
            batch_start = datetime.now()
            batch_gradients = None
            batch_loss = 0
            for x, target in zip(X_batch, y_batch):
                result = cvae.train_step(x, target, config)
                loss, latent_mean, latent_log, reconstructed_output, latent_vector, gradients = result

                batch_loss += loss
                if batch_gradients is None:
                    batch_gradients = gradients
                else:
                    for k, v in gradients.items():
                        batch_gradients[k] += v

            losses.append((len(X_batch), batch_loss))
            batch_loss /= len(X_batch)
            for k, v in batch_gradients.items():
                batch_gradients[k] /= len(X_batch)
            cvae.update_parameters(batch_gradients, config.learning_rate)

            logging.debug(f'''Batch {epoch + 1} - {batch_no}:
  Loss: {batch_loss}
  Duration: {datetime.now() - batch_start}''')

        average_loss = float(sum(x * y for x, y in losses)) / sum(x for x, _ in losses)
        logging.info(f'''Epoch {epoch + 1}:
  Average Loss: {average_loss}
  Duration: {datetime.now() - epoch_start}''')

def process_dataset(database, config):
    logging.info(f'Processing dataset ({config.start_row} -> {config.end_row}) ...')

    max_width = 0
    max_height = 0
    for i in range(config.start_row, config.end_row):
        H = database[i][4]
        max_width = max(max_width, H.shape[0])
        max_height = max(max_height, H.shape[1])

    features = []
    targets = []
    for i in range(config.start_row, config.end_row):
        _, _, E, _, H, _, _ = database[i]
        pad_width = max(0, max_width - H.shape[0])
        pad_height = max(0, max_height - H.shape[1])
        features.append(torch.from_numpy(np.pad(H, ((0, pad_width), (0, pad_height)))).to(device))
        targets.append(torch.from_numpy(E).to(device))

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=config.test_size,
        random_state=config.seed,
    )

    return (
        torch.stack(X_train, dim=0),
        torch.stack(X_test, dim=0),
        torch.stack(y_train, dim=0),
        torch.stack(y_test, dim=0),
    )

class nablaVAE:
    def __init__(self, config, input_shape):
        # Initialize network parameters based on input shape and latent dimension
        self.latent_dim = config.latent_dim

        # Encoder parameters
        self.encoder_fc1_weights = torch.randn(np.prod(input_shape), 64, device=device)
        self.encoder_fc2_weights_mean = torch.randn(64, self.latent_dim, device=device)
        self.encoder_fc2_weights_log_var = torch.randn(64, self.latent_dim, device=device)
        self.encoder_fc1_bias = torch.zeros(64, device=device)
        self.encoder_fc2_bias_mean = torch.zeros(self.latent_dim, device=device)
        self.encoder_fc2_bias_log_var = torch.zeros(self.latent_dim, device=device)

        # Decoder parameters
        self.decoder_fc1_weights = torch.randn(self.latent_dim, 64, device=device)
        self.decoder_fc1_bias = torch.zeros(64, device=device)

        # Output layers for the decoder
        self.decoder_fc2_weights = torch.randn(64, 128, device=device)
        self.decoder_fc3_weights = torch.randn(128, np.prod(input_shape), device=device)
        self.decoder_fc2_bias = torch.zeros(128, device=device)
        self.decoder_fc3_bias = torch.zeros(np.prod(input_shape), device=device)

        # Save input shape for later use
        self.input_shape = input_shape

    def train_step(self, x, target, config):
        # Forward pass--
        x = normalize(x)

        latent_mean, latent_log = self.encoder_forward(x, config)
        latent_vector = reparameterize(latent_mean, latent_log)

        reconstructed_output = self.decoder_forward(latent_vector, config)
        reconstructed_output = normalize(reconstructed_output)

        ## Compute reconstruction loss
        reconstruction_loss = torch.mean((x - reconstructed_output) ** 2)

        ## Compute KL divergence
        kl_divergence = -0.5 * torch.mean(1 + latent_log - latent_mean ** 2 - torch.exp(latent_log))
        energy_data = 0

        ## Total loss
        total_loss = reconstruction_loss + kl_divergence + energy_data

        # Backward pass and parameter update
        gradients = self.backward(
            x,
            latent_vector,
            latent_mean,
            latent_log,
            energy_data,
            config,
        )

        return (
            total_loss,
            latent_mean,
            latent_log,
            reconstructed_output,
            latent_vector,
            gradients,
        )

    def backward(
        self,
        x,
        latent_vector,
        latent_mean,
        latent_log,
        energy_data,
        config,
    ):
        # Initialize gradients
        gradients = {
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

        reconstruction_loss_gradient = -(x - self.decoder_forward(latent_vector, config))
        reconstruction_loss_gradient = reconstruction_loss_gradient.flatten()
        decoder_fc3_activation_gradient = reconstruction_loss_gradient

        # Compute gradient with respect to weights of decoder_fc3
        gradients['decoder_fc3_weights'] = (
            decoder_fc3_activation_gradient * self.decoder_fc3_weights
        ).sum(-1)

        # Compute gradient with respect to bias of decoder_fc3
        gradients['decoder_fc3_bias'] = torch.sum(decoder_fc3_activation_gradient, axis=0)

        reshaped_outputfc3 = self.pad_output_for_backward_pass(
            decoder_fc3_activation_gradient,
            self.decoder_fc2_weights.shape,
        )

        # Compute gradient of reconstruction loss with respect to activations of decoder_fc2
        decoder_fc2_activation_gradient = torch.matmul(reshaped_outputfc3, self.decoder_fc2_weights.T)
        decoder_fc2_activation_gradient *= leaky_relu_derivative(
            decoder_fc2_activation_gradient,
            config.leaky_relu_derivative_alpha,
        )

        # Compute gradient with respect to weights of decoder_fc2
        gradients['decoder_fc2_weights'] = torch.matmul(
            decoder_fc2_activation_gradient,
            self.decoder_fc2_weights,
        )

        # Compute gradient with respect to bias of decoder_fc2
        gradients['decoder_fc2_bias'] = torch.sum(decoder_fc2_activation_gradient, axis=0)

        # Compute gradient of reconstruction loss with respect to activations of decoder_fc1
        decoder_fc1_activation_gradient = torch.matmul(decoder_fc2_activation_gradient, self.decoder_fc1_weights.T)
        decoder_fc1_activation_gradient *= leaky_relu_derivative(
            decoder_fc1_activation_gradient,
            config.leaky_relu_derivative_alpha,
        )

        # Compute gradient with respect to weights of decoder_fc1
        gradients['decoder_fc1_weights'] = torch.matmul(decoder_fc1_activation_gradient, self.decoder_fc1_weights)

        # Compute gradient with respect to bias of decoder_fc1
        gradients['decoder_fc1_bias'] = torch.sum(decoder_fc1_activation_gradient, axis=0)

        # Compute gradient of KL divergence term
        kl_div_mean_gradient = 0.5 * (2 * latent_mean)
        kl_div_log_var_gradient = 0.5 * (1 - torch.exp(latent_log))

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
        gradients['encoder_fc2_weights_mean'] = torch.matmul(self.encoder_fc2_weights_mean, reshaped_gradients_mean)

        # Compute gradient with respect to bias of encoder_fc2_mean
        gradients['encoder_fc2_bias_mean'] = torch.sum(reshaped_gradients_mean, axis=0)

        # Compute gradient with respect to weights of encoder_fc2_log_var
        gradients['encoder_fc2_weights_log_var'] = torch.matmul(
            self.encoder_fc2_weights_log_var,
            reshaped_gradients_log_var,
        )

        ## Compute gradient with respect to bias of encoder_fc2_log_var
        gradients['encoder_fc2_bias_log_var'] = torch.sum(reshaped_gradients_log_var, axis=0)

        # Compute gradient of encoder_fc1
        encoder_fc1_activation_gradient = torch.matmul(
            reshaped_gradients_mean,
            self.encoder_fc1_weights.T,
        )
        encoder_fc1_activation_gradient += torch.matmul(
            reshaped_gradients_log_var, 
            self.encoder_fc1_weights.T,
        )
        encoder_fc1_activation_gradient *= leaky_relu_derivative(
            encoder_fc1_activation_gradient,
            config.leaky_relu_derivative_alpha,
        )

        # Compute gradient with respect to weights of encoder_fc1
        gradients['encoder_fc1_weights'] = torch.matmul(
            self.encoder_fc1_weights.T,
            encoder_fc1_activation_gradient.T,
        )

        # Compute gradient with respect to bias of encoder_fc1
        gradients['encoder_fc1_bias'] = torch.sum(encoder_fc1_activation_gradient, axis=0)

        return gradients

    def decoder_forward(self, latent_vector, config):
        # Forward pass through the decoder
        fc1_output = latent_vector @ self.decoder_fc1_weights + self.decoder_fc1_bias
        fc1_output = leaky_relu(fc1_output, config.leaky_relu_alpha)
        fc1_output = normalize(fc1_output)

        fc2_output = fc1_output @ self.decoder_fc2_weights + self.decoder_fc2_bias
        fc2_output = leaky_relu(fc2_output, config.leaky_relu_alpha)

        final = fc2_output @ self.decoder_fc3_weights + self.decoder_fc3_bias
        return final.reshape(self.input_shape)
    
    def encoder_forward(self, x, config):
        ## Flatten the output of the convolutional layers
        flattened = x.flatten()

        ## Forward pass through the fully connected layers
        fc1_output = flattened @ self.encoder_fc1_weights + self.encoder_fc1_bias
        fc1_output = leaky_relu(fc1_output, config.leaky_relu_alpha)

        latent_mean = fc1_output @ self.encoder_fc2_weights_mean + self.encoder_fc2_bias_mean
        latent_mean = leaky_relu(latent_mean, config.leaky_relu_alpha)
        latent_mean = normalize(latent_mean)

        latent_log = fc1_output @ self.encoder_fc2_weights_log_var + self.encoder_fc2_bias_log_var
        latent_log = leaky_relu(latent_log, config.leaky_relu_alpha)
        latent_log = normalize(latent_log)

        return latent_mean, latent_log

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

    def update_parameters(self, gradients, learning_rate):
        for k in gradients:
            value = getattr(self, k)
            setattr(self, k, value - learning_rate * value)

def batched(X, y, size):
    i = 0
    while i < len(X):
        yield X[i:i + size], y[i:i + size]
        i += size

def leaky_relu(x, alpha):
    return torch.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha):
    return torch.where(x > 0, 1, alpha)

def normalize(x):
    return x / torch.max(torch.abs(x))

def reparameterize(mu, log_var):
    epsilon = torch.randn(*mu.shape, device=device)
    return mu + torch.exp(0.5 * log_var) * epsilon

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
        end_row=1000,
        latent_dim=100,
        leaky_relu_alpha=0.2,
        leaky_relu_derivative_alpha=0.01,
        learning_rate=0.0001,
        num_epochs=25,
        seed=None,
        start_row=0,
        test_size=0.2,
        use_gpu=True,
    ):
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.end_row = end_row
        self.latent_dim = latent_dim
        self.leaky_relu_alpha = leaky_relu_alpha
        self.leaky_relu_derivative_alpha = leaky_relu_derivative_alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.seed = seed
        self.start_row = start_row
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

    config = None
    try:
        with open(args.config) as f:
            config = yaml.load(f, yaml.Loader)
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.warning(e)
    config = Config(**(config or {}))

    if config.use_gpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
    logging.info(f'device: {device}')

    if config.seed:
        torch.manual_seed(config.seed)

    main(config)
