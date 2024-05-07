#!/usr/bin/env python

from datetime import datetime
import json
import logging
import os
import sys

from nablaDFT.dataset import hamiltonian_dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import yaml

device = "cpu"
MODEL_DIR = 'models'

def main(config):
    database = hamiltonian_dataset.HamiltonianDatabase(config.dataset_path)
    X_train, X_test, y_train, y_test = process_dataset(database, config)
    logging.debug(f'Hamiltonian shape: {X_train[0].shape}')

    logging.info(f'Starting training ({len(X_train)} samples) ...')
    model, stats = train(X_train, y_train, config)

    if config.save:
        if config.model_path:
            output = config.model_path
        else:
            output = os.path.join(
                MODEL_DIR,
                f'{datetime.now().isoformat(timespec="seconds")}.pt',
            )
        ensure_directory(output)
        logging.info(f'Saving model to {output} ...')
        torch.save(model.__dict__, output)

    if args.json:
        print(json.dumps(stats))
    else:
        print('Final statistics:')
        print(f'  Loss: {stats["loss"]:.4f}')
        print(f'  Minimum Loss: {stats["minimum_loss"]:.4f}')
        print(f'  Reconstruction: {stats["reconstruction"]:.4f}')
        print(f'  KL Divergence: {stats["kl_divergence"]:.4f}')

def train(X, y, config):
    mean = torch.mean(X, axis=0).flatten()
    std = torch.std(X, axis=0).flatten()
    cvae = nablaVAE(config, X[0].shape, mean, std)
    maxv = torch.max(torch.max(torch.abs(X)), torch.max(torch.abs(y)))
    X /= maxv
    y /= maxv

    stats = {}
    for epoch in range(config.num_epochs):
        epoch_start = datetime.now()
        losses = []
        for batch_no, (X_batch, y_batch) in enumerate(batched(X, y, config.batch_size)):
            batch_start = datetime.now()
            batch_gradients = None
            batch_loss = 0
            batch_reconstruction = 0
            batch_kl_divergence = 0
            for x, target in zip(X_batch, y_batch):
                reconstruction, kl_divergence, energy, gradients = cvae.train_step(x, target)

                batch_loss += reconstruction + kl_divergence
                batch_reconstruction += reconstruction
                batch_kl_divergence += kl_divergence
                stats['loss'] = reconstruction + kl_divergence
                if 'minimum_loss' not in stats:
                    stats['minimum_loss'] = stats['loss']
                else:
                    stats['minimum_loss'] = min(stats['minimum_loss'], stats['loss'])
                stats['reconstruction'] = reconstruction
                stats['kl_divergence'] = kl_divergence
                if batch_gradients is None:
                    batch_gradients = gradients
                else:
                    for k, v in gradients.items():
                        batch_gradients[k] += v

            losses.append((len(X_batch), batch_loss))
            batch_loss /= len(X_batch)
            if 'max_batch_loss' not in stats:
                stats['max_batch_loss'] = batch_loss
            else:
                stats['max_batch_loss'] = max(stats['max_batch_loss'], batch_loss)

            for k, v in batch_gradients.items():
                batch_gradients[k] /= len(X_batch)
            cvae.update_parameters(batch_gradients)

            logging.debug(f'''Batch {epoch + 1} - {batch_no}:
  Loss: {stats["loss"]}
  Average Loss: {batch_loss}
  Average Reconstruction: {batch_reconstruction / len(X_batch)}
  Average KL Divergence: {batch_kl_divergence / len(X_batch)}
  Duration: {datetime.now() - batch_start}''')

        loss = float(sum(x * y for x, y in losses)) / sum(x for x, _ in losses)
        logging.info(f'''Epoch {epoch + 1}:
  Loss: {stats["loss"]}
  Average Loss: {loss}
  Duration: {datetime.now() - epoch_start}''')

    for k in stats:
        stats[k] = float(stats[k].detach().numpy())

    return cvae, stats

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
    def __init__(self, config, input_shape, mean, std):
        # Initialize network parameters based on input shape and latent dimension
        self.clip = config.clip
        self.input_shape = input_shape
        self.latent_dim = config.latent_dim
        self.leaky_relu_alpha = config.leaky_relu_alpha
        self.learning_rate = config.learning_rate
        self.mean = mean
        self.std = std

        # Encoder parameters
        self.encoder_fc1_weights = randn(np.prod(input_shape), config.encoder_fc1_size)
        self.encoder_fc1_bias = zeros(config.encoder_fc1_size)
        self.encoder_fc2_weights_mean = randn(config.encoder_fc1_size, self.latent_dim)
        self.encoder_fc2_bias_mean = zeros(self.latent_dim)
        self.encoder_fc2_weights_log_var = randn(config.encoder_fc1_size, self.latent_dim)
        self.encoder_fc2_bias_log_var = zeros(self.latent_dim)

        # Decoder parameters
        self.decoder_fc1_weights = randn(self.latent_dim, config.decoder_fc1_size)
        self.decoder_fc1_bias = zeros(config.decoder_fc1_size)
        self.decoder_fc2_weights = randn(config.decoder_fc1_size, config.decoder_fc2_size)
        self.decoder_fc2_bias = zeros(config.decoder_fc2_size)
        self.decoder_fc3_weights = randn(config.decoder_fc2_size, np.prod(input_shape))
        self.decoder_fc3_bias = zeros(np.prod(input_shape))

    @classmethod
    def load(cls, config):
        cvae = cls(config, 1, None, None)
        cvae.__dict__ = torch.load(config.model_path)
        return cvae

    def train_step(self, x, target):
        # Forward pass
        activations = {}
        self.encoder_forward(x, activations)
        latent_mean, latent_log = activations['latent_mean'], activations['latent_log']
        latent_vector = reparameterize(latent_mean, latent_log)

        self.decoder_forward(latent_vector, activations)
        reconstructed_output = activations['decoder_fc3'].reshape(self.input_shape)

        ## Compute reconstruction loss
        reconstruction_loss = torch.mean((x - reconstructed_output) ** 2)
 
        ## Compute KL divergence
        kl_divergence = -0.5 * torch.sum(1 + latent_log - latent_mean ** 2 - torch.exp(latent_log))
        energy_data = 0
        
        # Backward pass and parameter update
        gradients = self.backward(
            x,
            reconstruction_loss,
            kl_divergence,
            latent_vector,
            latent_mean,
            latent_log,
            energy_data,
            activations,
        )

        return reconstruction_loss, kl_divergence, energy_data, gradients

    def backward(
        self,
        x,
        reconstruction_loss,
        kl_divergence,
        latent_vector,
        latent_mean,
        latent_log,
        energy_data,
        activations,
    ):
        gradients = { 'decoder_fc3_weights': -(x.flatten() - activations['decoder_fc3'])  * activations['decoder_fc3'] }

        t = gradients['decoder_fc3_weights'] @ self.decoder_fc3_weights.T
        gradients['decoder_fc2_weights'] = t * self.leaky_relu_derivative(activations['decoder_fc2'])

        t = gradients['decoder_fc2_weights'] @ self.decoder_fc2_weights.T
        gradients['decoder_fc1_weights'] = t * self.leaky_relu_derivative(activations['decoder_fc1'])

        t = gradients['decoder_fc1_weights'] @ self.decoder_fc1_weights.T
        gradients['encoder_fc2_weights_mean'] = t * self.leaky_relu_derivative(
            activations['latent_mean'],
        )
        kl_div_mean = 0.5 * (2 * latent_mean)
        gradients['encoder_fc2_weights_mean'] += kl_div_mean

        t = gradients['decoder_fc1_weights'] @ self.decoder_fc1_weights.T
        gradients['encoder_fc2_weights_log_var'] = t * self.leaky_relu_derivative(
            activations['latent_log'],
        )
        kl_div_log_var = 0.5 * (1 - torch.exp(latent_log))
        gradients['encoder_fc2_weights_log_var'] += kl_div_log_var

        t = gradients['encoder_fc2_weights_mean'] @ self.encoder_fc2_weights_mean.T
        gradients['encoder_fc1_weights'] = t * self.leaky_relu_derivative(activations['encoder_fc1'])

        if self.clip:
            total_norm = torch.norm(torch.stack([
                torch.norm(grad.detach())
                for grad in gradients.values()
            ]))

            if total_norm > self.clip:
                clip_coefficient = self.clip / (total_norm + 1e-6)
                for k in gradients:
                    gradients[k] *= clip_coefficient

        return gradients

    def decoder_forward(self, latent_vector, activations):
        # Forward pass through the decoder
        fc1_output = latent_vector @ self.decoder_fc1_weights + self.decoder_fc1_bias
        fc1_output = normalize(self.leaky_relu(fc1_output))
        activations['decoder_fc1'] = fc1_output

        fc2_output = fc1_output @ self.decoder_fc2_weights + self.decoder_fc2_bias
        fc2_output = normalize(self.leaky_relu(fc2_output))
        activations['decoder_fc2'] = fc2_output

        fc3_output = fc2_output @ self.decoder_fc3_weights + self.decoder_fc3_bias
        fc3_output = normalize(fc3_output)
        if self.clip:
            fc3_output = fc3_output * self.std + self.mean
        activations['decoder_fc3'] = fc3_output
        
    def encoder_forward(self, x, activations):
        flattened = x.flatten()

        fc1_output = flattened @ self.encoder_fc1_weights + self.encoder_fc1_bias
        fc1_output = normalize(self.leaky_relu(fc1_output))
        activations['encoder_fc1'] = fc1_output

        latent_mean = fc1_output @ self.encoder_fc2_weights_mean + self.encoder_fc2_bias_mean
        latent_mean = normalize(self.leaky_relu(latent_mean))
        activations['latent_mean'] = latent_mean

        latent_log = fc1_output @ self.encoder_fc2_weights_log_var + self.encoder_fc2_bias_log_var
        latent_log = normalize(self.leaky_relu(latent_log))
        activations['latent_log'] = latent_log

    def generate(self):
        activations = {}
        latent_vector = randn(1, self.latent_dim)
        self.decoder_forward(latent_vector, activations)
        return activations['decoder_fc3'].reshape(self.input_shape)

    def leaky_relu(self, x):
        return torch.where(x > 0, x, self.leaky_relu_alpha * x)

    def leaky_relu_derivative(self, x):
        return torch.where(x > 0, 1, self.leaky_relu_alpha)

    def update_parameters(self, gradients):
        for k in gradients:
            setattr(self, k, getattr(self, k) - self.learning_rate * gradients[k])

def batched(X, y, size):
    i = 0
    while i < len(X):
        yield X[i:i + size], y[i:i + size]
        i += size

def ensure_directory(path):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

def normalize(x):
    return x / torch.max(torch.abs(x))

def randn(*args, **kwargs):
    return torch.randn(*args, device=device, **kwargs)

def reparameterize(mu, log_var):
    epsilon = randn(*mu.shape)
    return mu + torch.exp(0.5 * log_var) * epsilon

def zeros(*args, **kwargs):
    return torch.zeros(*args, device=device, **kwargs)

class Config:
    def __init__(
        self,
        *,
        batch_size=32,
        clip=None,
        dataset_path='dataset_train_2k.db',
        decoder_fc1_size=64,
        decoder_fc2_size=128,
        encoder_fc1_size=64,
        end_row=2000,
        latent_dim=100,
        leaky_relu_alpha=0.2,
        learning_rate=0.00001,
        model_path=None,
        num_epochs=25,
        save=False,
        seed=None,
        start_row=0,
        test_size=0.2,
        use_gpu=True,
    ):
        self.batch_size = batch_size
        self.clip = clip
        self.dataset_path = dataset_path
        self.decoder_fc1_size = decoder_fc1_size
        self.decoder_fc2_size = decoder_fc2_size
        self.encoder_fc1_size = encoder_fc1_size
        self.end_row = end_row
        self.latent_dim = latent_dim
        self.leaky_relu_alpha = leaky_relu_alpha
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.num_epochs = num_epochs
        self.save = save
        self.seed = seed
        self.start_row = start_row
        self.test_size = test_size
        self.use_gpu = use_gpu

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='main.yaml')
    parser.add_argument('-g', '--generate', metavar='FILE')
    parser.add_argument('-j', '--json', action='store_true', help='output statistics as json')
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

    if args.generate:
        if not config.model_path:
            raise RuntimeError(
                '`model_path` must be set in configuration file in order to generate hamiltonian data',
            )

        cvae = nablaVAE.load(config)
        tensor = cvae.generate()
        logging.info(f'Saving generated tensor to {args.generate} ...')
        ensure_directory(args.generate)
        torch.save(tensor, args.generate)
    else:
        main(config)
