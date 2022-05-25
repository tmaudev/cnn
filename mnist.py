import numpy as np
from keras.datasets import mnist

class Layer:
    input_size = None

    def __init__(self, input_size):
        self.input_size = input_size


class ConvLayer(Layer):
    weights = None
    input_size = None
    stride = None
    padding = None
    output = None
    output_size = None

    def __init__(self, input_size, filter_size, num_filters, stride, padding):
        self.input_size = input_size
        self.stride = stride
        self.padding = padding

        c, h, w = input_size
        f = filter_size
        self.weights = np.random.default_rng().uniform(low=-0.1, high=0.1, size=(num_filters, f, f, c))
        self.output_size = int((w - f + padding * 2) / stride + 1)

    # Tensor is size (c, h, w)
    def forwardPass(self, tensor):
        num_filters = self.weights.shape[0]
        self.output = np.empty((self.output_size, self.output_size, num_filters))
        f = self.weights.shape[1]
        for i in range(self.output_size):
            for j in range(self.output_size):
                for w in range(num_filters):
                    input_window = tensor[i:i+f, j:j+f, :]
                    assert(input_window.shape == self.weights[w].shape)
                    mult = np.multiply(input_window, self.weights[w])
                    self.output[i][j][w] = np.maximum(0, np.sum(mult))

class FCLayer(Layer):
    weights = None
    num_nodes = None
    output = None

    def __init__(self, input_size, num_nodes):
        self.num_nodes = num_nodes

        c, h, w = input_size
        self.weights = np.random.default_rng().uniform(low=-0.1, high=0.1, size=(c * h * w, num_nodes))

    # Tensor is size (c, h, w)
    def forwardPass(self, tensor):
        flattened = tensor.flatten()
        product = np.dot(flattened, self.weights)
        exp = np.exp(product)
        self.output = exp / np.sum(exp)


class MnistCNN:
    layers = []

    def addConvLayer(self, input_size, filter_size, num_filters, stride, padding):
        layer = ConvLayer(input_size, filter_size, num_filters, stride, padding)
        self.layers.append(layer)
        return (num_filters, layer.output_size, layer.output_size)

    def addFCLayer(self, input_size, num_nodes):
        self.layers.append(FCLayer(input_size, num_nodes))

    def train(self, train_x, train_y, batch_size, epochs, learning_rate):
        for epoch in range(epochs):
            print("Epoch ", epoch)
            for example in train_x:
                layer_input = np.expand_dims(example, axis=2)
                for layer in self.layers:
                    layer.forwardPass(layer_input)
                    layer_input = layer.output


if __name__ == '__main__':
    print("Loading dataset...")
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    num_examples, h, w = train_x.shape
    c = 1
    num_classes = 10

    cnn = MnistCNN()
    c, h, w = cnn.addConvLayer((c, h, w), 3, 10, 1, 0)
    c, h, w = cnn.addConvLayer((c, h, w), 3, 10, 1, 0)
    c, h, w = cnn.addConvLayer((c, h, w), 3, 10, 1, 0)
    cnn.addFCLayer((c, h, w), num_classes)

    cnn.train(train_x, train_y, 10, 1, 1e-5)
