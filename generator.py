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
import cv2
import numpy as np

from concurrent.futures.thread import ThreadPoolExecutor


class DataGenerator:
    def __init__(self,
                 image_paths,
                 generate_shape,
                 batch_size,
                 dtype='float32'):
        self.image_paths = image_paths
        self.generate_shape = generate_shape
        self.batch_size = batch_size
        self.dtype = dtype
        self.pool = ThreadPoolExecutor(8)
        self.img_index = 0
        np.random.shuffle(self.image_paths)

    def load(self):
        fs = []
        for _ in range(self.batch_size):
            fs.append(self.pool.submit(self.load_image, self.next_image_path()))
        ae_x = []
        for f in fs:
            img = f.result()
            x = self.preprocess(img)
            ae_x.append(x)
        ae_x = np.asarray(ae_x).astype(self.dtype)
        return ae_x

    @staticmethod
    def denormalize(x):
        return np.asarray(np.clip((x * 255.0), 0.0, 255.0)).astype('uint8')

    @staticmethod
    def get_z_vector(size):
        return np.random.normal(loc=0.0, scale=1.0, size=size)

    def preprocess(self, img):
        img = self.resize(img, (self.generate_shape[1], self.generate_shape[0]))
        x = np.asarray(img).reshape(self.generate_shape).astype('float32') / 255.0
        return x

    def next_image_path(self):
        path = self.image_paths[self.img_index]
        self.img_index += 1
        if self.img_index == len(self.image_paths):
            self.img_index = 0
            np.random.shuffle(self.image_paths)
        return path

    def resize(self, img, size):
        interpolation = None
        img_height, img_width = img.shape[:2]
        if size[0] > img_width or size[1] > img_height:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_AREA
        return cv2.resize(img, size, interpolation=interpolation)

    def load_image(self, image_path):
        return cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE if self.generate_shape[-1] == 1 else cv2.IMREAD_COLOR)

