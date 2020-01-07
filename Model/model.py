from tensorflow.keras import models, layers

class Softmax:
    def build(shape):
        model = models.Sequential()
        model.add(layers.Dense(2, activation='softmax', input_shape=shape))
        return model
