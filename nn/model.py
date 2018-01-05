import numpy as np
import os
from . import ConvNetA
from uuid import uuid1
from PIL import Image
from keras import backend as K
K.set_image_dim_ordering('th')


class Classifier:

    def __init__(self):
        self.cnn_a = ConvNetA()
        self.cnn_a.setup_weights()

    @staticmethod
    def refine_img(image):
        filename = str(uuid1()) + '.png'

        with open('tmp/' + filename, 'wb') as file:
            file.write(image)

        plain_image = Image.open('tmp/' + filename)

        # Resize and turn into greyscale
        img = plain_image.resize((28, 28), Image.NEAREST).convert('L', colors=255)

        # Normalize and inverse
        img = (255.0 - np.array(img)) / 255.0

        # Reshape
        img = np.array(img).reshape(-1, 1, 28, 28)
        os.remove('tmp/' + filename)
        return img

    def predict(self, image):
        img = self.refine_img(image)
        print(np.sum(img))
        if np.sum(img) <= .5:
            return "Draw something please :)"
        cnn_a_out = self.cnn_a.predict(img)
        return {'cnn_a': cnn_a_out}
