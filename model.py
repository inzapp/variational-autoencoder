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
import tensorflow as tf


class Model:
    def __init__(self, generate_shape, latent_dim):
        self.generate_shape = generate_shape
        self.latent_dim = latent_dim
        self.vae = None
        self.vae_e = None
        self.vae_d = None
        self.strides = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.filters = [16, 32, 64, 128, 256, 512, 512, 512, 512, 512]
        self.stride = self.calc_stride(self.generate_shape)
        self.stride_index = self.strides.index(self.stride)
        self.latent_rows = generate_shape[0] // self.stride
        self.latent_cols = generate_shape[1] // self.stride
        self.latent_channels = self.filters[self.stride_index-1]

    def calc_stride(self, generate_shape):
        stride = 32
        min_size = min(generate_shape[:2])
        for v in self.strides:
            if min_size >= v and min_size % v == 0:
                stride = v
            else:
                break
        return stride

    def build(self, vae_d=None, gap=False):
        assert self.generate_shape[0] % 32 == 0 and self.generate_shape[1] % 32 == 0
        ae_e_input, mu, log_var, ae_e_output = self.build_ae_e(bn=False, gap=gap)
        self.vae_e = tf.keras.models.Model(ae_e_input, ae_e_output)
        if vae_d is None:
            ae_d_input, ae_d_output = self.build_ae_d(bn=False, gap=gap)
            self.vae_d = tf.keras.models.Model(ae_d_input, ae_d_output)
        else:
            ae_d_input, ae_d_output = vae_d.input, vae_d.output
            self.vae_d = vae_d

        ae_output = self.vae_d(ae_e_output)
        self.vae = tf.keras.models.Model(ae_e_input, [mu, log_var, ae_output])
        return self.vae, self.vae_e, self.vae_d

    def build_ae_e(self, bn, gap):
        ae_e_input = tf.keras.layers.Input(shape=self.generate_shape)
        x = ae_e_input
        for i in range(self.stride_index):
            x = self.conv2d(x, self.filters[i], 5, 2, activation='leaky', bn=bn)
        if gap:
            mu = self.gap2d(self.conv2d(x, self.latent_dim, 1, 1, activation='linear', bn=True))
            log_var = self.gap2d(self.conv2d(x, self.latent_dim, 1, 1, activation='linear', bn=True))
            ae_e_output = self.sampling(mu, log_var)
        else:
            x = self.flatten(x)
            mu = self.dense(x, self.latent_dim, activation='linear', bn=True)
            log_var = self.dense(x, self.latent_dim, activation='linear', bn=True)
            ae_e_output = self.sampling(mu, log_var)
        return ae_e_input, mu, log_var, ae_e_output

    def build_ae_d(self, bn, gap):
        ae_d_input = tf.keras.layers.Input(shape=(self.latent_dim,))
        x = ae_d_input
        if gap:
            x = self.dense(x, self.latent_rows * self.latent_cols * self.latent_dim, activation='leaky', bn=bn)
            x = self.reshape(x, (self.latent_rows, self.latent_cols, self.latent_dim))
            x = self.conv2d(x, self.latent_channels, 1, 1, activation='leaky', bn=bn)
        else:
            x = self.dense(x, self.latent_rows * self.latent_cols * self.latent_channels, activation='leaky', bn=bn)
            x = self.reshape(x, (self.latent_rows, self.latent_cols, self.latent_channels))
        for i in range(self.stride_index-1, -1, -1):
            x = self.conv2d_transpose(x, self.filters[i], 4, 2, activation='leaky', bn=bn)
        ae_d_output = self.conv2d_transpose(x, self.generate_shape[-1], 1, 1, activation='sigmoid')
        return ae_d_input, ae_d_output

    def sampling(self, mu, log_var):
        def function(args):
            mu, log_var = args
            epsilon = tf.random.normal(shape=tf.shape(mu))
            return mu + tf.exp(log_var * 0.5) * epsilon
        return tf.keras.layers.Lambda(function=function)([mu, log_var])

    def conv2d(self, x, filters, kernel_size, strides, bn=False, activation='leaky'):
        x = tf.keras.layers.Conv2D(
            strides=strides,
            filters=filters,
            padding='same',
            use_bias=not bn,
            kernel_size=kernel_size,
            kernel_regularizer=self.kernel_regularizer(),
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def conv2d_transpose(self, x, filters, kernel_size, strides, bn=False, activation='leaky'):
        x = tf.keras.layers.Conv2DTranspose(
            strides=strides,
            filters=filters,
            padding='same',
            use_bias=not bn,
            kernel_size=kernel_size,
            kernel_regularizer=self.kernel_regularizer(),
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def dense(self, x, units, bn=False, activation='leaky'):
        x = tf.keras.layers.Dense(
            units=units,
            use_bias=not bn,
            kernel_initializer=self.kernel_initializer())(x)
        if bn:
            x = self.batch_normalization(x)
        return self.activation(x, activation)

    def gap2d(self, x):
        return tf.keras.layers.GlobalAveragePooling2D()(x)

    def batch_normalization(self, x):
        return tf.keras.layers.BatchNormalization(momentum=0.9)(x)

    def kernel_initializer(self):
        return 'he_normal'

    def kernel_regularizer(self, l2=0.01):
        return tf.keras.regularizers.l2(l2=l2)

    def activation(self, x, activation):
        if activation == 'leaky':
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        elif activation != 'linear':
            x = tf.keras.layers.Activation(activation=activation)(x)
        return x

    def reshape(self, x, target_shape):
        return tf.keras.layers.Reshape(target_shape=target_shape)(x)

    def flatten(self, x):
        return tf.keras.layers.Flatten()(x)

    def summary(self):
        self.vae.summary()
        print()
        self.vae_e.summary()
        print()
        self.vae_d.summary()

