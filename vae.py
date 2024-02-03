"""
Authors : inzapp

Github url : https://github.com/inzapp/variational-autoencoder

Copyright (c) 2024 Inzapp

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import cv2
import warnings
import numpy as np
import silence_tensorflow.auto
import tensorflow as tf

from glob import glob
from tqdm import tqdm
from time import time
from model import Model
from eta import ETACalculator
from generator import DataGenerator
from lr_scheduler import LRScheduler
from ckpt_manager import CheckpointManager


class TrainingConfig:
    def __init__(self,
                 train_image_path,
                 generate_shape,
                 lr,
                 batch_size,
                 latent_dim,
                 save_interval,
                 iterations,
                 view_grid_size,
                 model_name,
                 pretrained_vae_d_path='',
                 training_view=False):
        self.train_image_path = train_image_path
        self.generate_shape = generate_shape
        self.lr = lr
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.save_interval = save_interval
        self.iterations = iterations
        self.view_grid_size = view_grid_size
        self.model_name = model_name
        self.pretrained_vae_d_path = pretrained_vae_d_path
        self.training_view = training_view


class VariationalAutoencoder(CheckpointManager):
    def __init__(self, config):
        super().__init__()
        assert config.generate_shape[0] % 32 == 0
        assert config.generate_shape[1] % 32 == 0
        assert config.generate_shape[2] in [1, 3]
        self.train_image_path = config.train_image_path
        self.generate_shape = config.generate_shape
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.latent_dim = config.latent_dim
        self.save_interval = config.save_interval
        self.iterations = config.iterations
        self.view_grid_size = config.view_grid_size
        self.model_name = config.model_name
        self.pretrained_vae_d_path = config.pretrained_vae_d_path
        self.training_view = config.training_view

        self.decoding_params = None
        self.set_model_name(self.model_name)
        self.live_view_previous_time = time()
        warnings.filterwarnings(action='ignore')

        if self.pretrained_vae_d_path == '':
            self.model = Model(generate_shape=self.generate_shape, latent_dim=self.latent_dim)
            self.vae, self.vae_e, self.vae_d = self.model.build()
        else:
            pretrained_ae_d = None
            if self.pretrained_vae_d_path != '':
                if os.path.exists(self.pretrained_vae_d_path) and os.path.isfile(self.pretrained_vae_d_path):
                    pretrained_ae_d = tf.keras.models.load_model(self.pretrained_vae_d_path, compile=False)
                    self.vae_d = pretrained_ae_d
                    self.generate_shape = self.vae_d.output_shape[1:]
                    self.latent_dim = pretrained_ae_d.input.shape[-1]
                else:
                    print(f'decoder file not found : {self.pretrained_vae_d_path}')
                    exit(0)

            self.model = Model(generate_shape=self.generate_shape, latent_dim=self.latent_dim)
            self.vae, self.vae_e, self.vae_d = self.model.build(vae_d=pretrained_ae_d)

        self.train_image_paths = self.init_image_paths(self.train_image_path)
        self.train_data_generator = DataGenerator(
            image_paths=self.train_image_paths,
            generate_shape=self.generate_shape,
            batch_size=self.batch_size)

    def init_image_paths(self, image_path):
        return glob(f'{image_path}/**/*.jpg', recursive=True)

    def compute_gradient(self, model, optimizer, x):
        with tf.GradientTape() as tape:
            mu, log_var, y_pred = model(x, training=True)
            mu_mean = tf.reduce_mean(mu)
            log_var_mean = tf.reduce_mean(log_var)
            reconstruction_mse = tf.reduce_mean(tf.square(x - y_pred))
            kld = -0.5 * (1.0 + log_var - tf.square(mu) - tf.exp(log_var))
            kld_mean = tf.reduce_mean(kld)
            kld_sum = tf.reduce_sum(kld)
            batch_size_f = tf.cast(tf.shape(x)[0], dtype=y_pred.dtype)
            loss = reconstruction_mse + ((kld_sum * 0.001) / batch_size_f)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return reconstruction_mse, kld_mean, mu_mean, log_var_mean

    def print_loss(self, progress_str, loss_vars):
        mse, kld, mu, log_var = loss_vars
        loss_str = f'\r{progress_str}'
        loss_str += f' reconstruction_loss : {mse:>8.4f}'
        loss_str += f', kl_divergence : {kld:>8.4f}'
        loss_str += f', mu : {mu:>8.4f}'
        loss_str += f', log_var : {log_var:>8.4f}'
        print(loss_str, end='')

    def train(self):
        if len(self.train_image_paths) == 0:
            print(f'no images found in {self.train_image_path}')
            exit(0)

        self.model.summary()
        print(f'\ntrain on {len(self.train_image_paths)} samples.')
        print('start training')
        iteration_count = 0
        compute_gradient = tf.function(self.compute_gradient)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        lr_scheduler = LRScheduler(lr=self.lr, iterations=self.iterations, warm_up=0.0, policy='constant')
        self.init_checkpoint_dir()
        eta_calculator = ETACalculator(iterations=self.iterations)
        eta_calculator.start()
        while True:
            ae_x = self.train_data_generator.load()
            lr_scheduler.update(optimizer, iteration_count)
            loss_vars = compute_gradient(self.vae, optimizer, ae_x)
            iteration_count += 1
            progress_str = eta_calculator.update(iteration_count)
            self.print_loss(progress_str, loss_vars)
            if self.training_view:
                self.training_view_function()
            if iteration_count % self.save_interval == 0:
                model_path_without_extention = f'{self.checkpoint_path}/model_{iteration_count}_iter'
                self.vae.save(f'{model_path_without_extention}_vae.h5', include_optimizer=False)
                self.vae_d.save(f'{model_path_without_extention}_vae_d.h5', include_optimizer=False)
                # self.decoding_params = self.save_decoding_params(f'{model_path_without_extention}_decoding_params.txt')
                generated_images = self.generate_image_grid(grid_size=21 if self.latent_dim == 2 else 10)
                cv2.imwrite(f'{model_path_without_extention}.jpg', generated_images)
                print(f'\nsave success : {model_path_without_extention}\n')
            if iteration_count == self.iterations:
                if self.latent_dim == 2:
                    self.visualize_latent_vectors(mode='true')
                    self.visualize_latent_vectors(mode='pred')
                print('\ntrain end successfully')
                exit(0)

    def save_decoding_params(self, save_path, sample_size=500):
        latent_vectors = []
        sample_size = len(self.train_image_paths) if sample_size == 'max' else sample_size
        np.random.shuffle(self.train_image_paths)
        for path in tqdm(self.train_image_paths[:sample_size]):
            img = self.train_data_generator.load_image(path)
            x = self.train_data_generator.preprocess(img)
            latent_vector = np.asarray(self.graph_forward(self.vae_e, x.reshape((1,) + x.shape))).reshape((self.latent_dim,))
            latent_vectors.append(latent_vector)
        latent_vectors = np.asarray(latent_vectors).reshape((sample_size, self.latent_dim)).astype('float32')
        lv_mean = np.mean(latent_vectors, axis=0)
        lv_std = np.std(latent_vectors, axis=0)
        lv_min = np.min(latent_vectors, axis=0)
        lv_max = np.max(latent_vectors, axis=0)
        decoding_params = []
        decoding_param_str = ''
        for i in range(self.latent_dim):
            decoding_param = [lv_mean[i], lv_std[i], lv_min[i], lv_max[i]]
            decoding_params.append(decoding_param)
            decoding_param_str += f'{lv_mean[i]:.6f} {lv_std[i]:.6f} {lv_min[i]:.6f} {lv_max[i]:.6f}\n'
        with open(save_path, 'wt') as f:
            f.writelines(decoding_param_str)
        return decoding_params

    def visualize_latent_vectors(self, mode):
        from matplotlib import pyplot as plt
        assert self.latent_dim == 2
        assert mode in ['true', 'pred']
        x_datas = []
        y_datas = []
        for path in tqdm(self.train_image_paths):
            if mode == 'true':
                latent_vector = self.train_data_generator.get_z_vector(size=self.latent_dim)
            else:
                img = self.train_data_generator.load_image(path)
                x = self.train_data_generator.preprocess(img)
                latent_vector = np.asarray(self.graph_forward(self.vae_e, x.reshape((1,) + x.shape))).reshape((self.latent_dim,))
            x_datas.append(latent_vector[0])
            y_datas.append(latent_vector[1])
        plt.scatter(x_datas, y_datas)
        plt.tight_layout(pad=0.5)
        plt.savefig(f'{self.checkpoint_path}/latent_vector_{mode}.png')

    @staticmethod
    @tf.function
    def graph_forward(model, x):
        return model(x, training=False)

    def generate(self, save_count, save_grid_image, grid_size):
        if save_count > 0:
            save_dir_path = 'generated_images'
            os.makedirs(save_dir_path, exist_ok=True)
            elements = [chr(i) for i in range(48, 58)] + [chr(i) for i in range(65, 91)] + [chr(i) for i in range(97, 123)]
            for i in tqdm(range(save_count)):
                random_stamp = ''.join(np.random.choice(elements, 12))
                if save_grid_image:
                    save_path = f'{save_dir_path}/generated_grid_{i}_{random_stamp}.jpg'
                    img = self.generate_image_grid(grid_size=grid_size)
                else:
                    save_path = f'{save_dir_path}/generated_{i}_{random_stamp}.jpg'
                    img = self.generate_random_image(size=1)
                cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        else:
            while True:
                if save_grid_image:
                    img = self.generate_image_grid(grid_size=grid_size)
                else:
                    img = self.generate_random_image()
                cv2.imshow('img', img)
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)

    def generate_random_image(self, size=1):
        if self.decoding_params is None:
            z = np.asarray([DataGenerator.get_z_vector(size=self.latent_dim) for _ in range(size)])
        else:
            z = []
            for _ in range(size):
                latent_vector = np.zeros(shape=(self.latent_dim,), dtype=np.float32)
                for i in range(self.latent_dim):
                    lv_mean = self.decoding_params[i][0]
                    lv_min = self.decoding_params[i][2]
                    lv_max = self.decoding_params[i][3]
                    lv_min += (lv_mean - lv_min) * 0.9
                    lv_max += (lv_mean - lv_max) * 0.9
                    latent_vector[i] = np.random.uniform(low=lv_min, high=lv_max)

                    # lv_mean = self.decoding_params[i][0]
                    # lv_std = self.decoding_params[i][1] * 0.9
                    # latent_vector[i] = np.random.normal(loc=lv_mean, scale=lv_std)
                z.append(latent_vector)
            z = np.asarray(z)
        y = np.asarray(self.graph_forward(self.vae_d, z))
        generated_images = DataGenerator.denormalize(y).reshape((size,) + self.generate_shape)
        return generated_images[0] if size == 1 else generated_images

    def generate_latent_space_2d(self, split_size):
        assert split_size > 1
        assert self.latent_dim == 2
        if self.decoding_params is None:
            space_x = np.linspace(-2.0, 2.0, split_size)
            space_y = np.linspace(-2.0, 2.0, split_size)
        else:
            space_x = np.linspace(self.decoding_params[0][2], self.decoding_params[0][3], split_size)
            space_y = np.linspace(self.decoding_params[1][2], self.decoding_params[1][3], split_size)
        z = []
        for i in range(split_size):
            for j in range(split_size):
                z.append([space_y[i], space_x[j]])
        z = np.asarray(z).reshape((split_size * split_size, 2)).astype('float32')
        y = np.asarray(self.graph_forward(self.vae_d, z))
        y = DataGenerator.denormalize(y)
        generated_images = np.clip(np.asarray(y).reshape((split_size * split_size,) + self.generate_shape), 0.0, 255.0).astype('uint8')
        return generated_images

    def make_border(self, img, size=5):
        return cv2.copyMakeBorder(img, size, size, size, size, None, value=(192, 192, 192)) 

    def training_view_function(self):
        cur_time = time()
        if cur_time - self.live_view_previous_time > 3.0:
            generated_images = self.generate_image_grid(grid_size=self.view_grid_size)
            cv2.imshow('generated_images', generated_images)
            cv2.waitKey(1)
            self.live_view_previous_time = cur_time

    def generate_image_grid(self, grid_size):
        if grid_size == 'auto':
            border_size = 10
            grid_size = min(720 // (self.generate_shape[0] + border_size), 1280 // (self.generate_shape[1] + border_size))
        else:
            if type(grid_size) is str:
                grid_size = int(grid_size)
        if self.latent_dim == 2:
            generated_images = self.generate_latent_space_2d(split_size=grid_size)
        else:
            generated_images = self.generate_random_image(size=grid_size * grid_size)
        generated_image_grid = None
        for i in range(grid_size):
            grid_row = None
            for j in range(grid_size):
                generated_image = self.make_border(generated_images[i*grid_size+j])
                if grid_row is None:
                    grid_row = generated_image
                else:
                    grid_row = np.append(grid_row, generated_image, axis=1)
            if generated_image_grid is None:
                generated_image_grid = grid_row
            else:
                generated_image_grid = np.append(generated_image_grid, grid_row, axis=0)
        return generated_image_grid

