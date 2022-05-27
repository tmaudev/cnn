import numpy as np
import math
import matplotlib.pyplot as plt
from random import randrange
from keras.datasets import mnist

class Layer:
    input_size = None

    def __init__(self, input_size):
        self.input_size = input_size

class ConvLayer(Layer):
    weights = None
    biases = None
    output_size = None
    layer_output = None
    layer_input = None

    def __init__(self, input_size, filter_size, num_filters, stride, padding):
        self.stride = stride
        self.padding = padding

        c, h, w = input_size
        f = filter_size

        limit = 1 / math.sqrt(100)

        self.weights = np.random.uniform(low=-limit, high=limit, size=(f, f, c, num_filters))
        output_h = int((h - f + padding * 2) / stride + 1)
        output_w = int((w - f + padding * 2) / stride + 1)
        self.output_size = (output_h, output_w, num_filters)

        self.biases = np.zeros(self.output_size)

    def crossCorrelate(self, tensor, kernels, stride=1, padding=0, full=False):
        n, h, w, c = tensor.shape
        f = kernels.shape[0]
        num_filters = kernels.shape[3]

        if full:
            stride = 1
            padding = int(f / 2)

        output_h = int((h - f + padding * 2) / stride + 1)
        output_w = int((w - f + padding * 2) / stride + 1)

        tensor = np.pad(tensor, ((0,0), (padding, padding), (padding, padding), (0, 0)))

        output = np.empty((n, output_h, output_w, num_filters))
        for i in range(0, output_h, stride):
            for j in range(0, output_w, stride):
                window = tensor[:, i:i+f, j:j+f, :, np.newaxis]
                product = window * kernels[np.newaxis, :, :, :, :]
                sum_of_prod = np.sum(product, axis=(1, 2, 3))
                output[:, i, j, :] = sum_of_prod
        return output

    def conv2d(self, tensor, kernels, stride=1, padding=0, full=False):
        kernels = np.rot90(kernels, axes=(1,2))
        kernels = np.rot90(kernels, axes=(1,2))
        return self.crossCorrelate(tensor, kernels, stride, padding, full)

    # Tensor is size (n, c, h, w)
    def calculateOutput(self, tensor, training):
        if training:
            self.layer_input = tensor

        output = self.crossCorrelate(tensor, self.weights, self.stride, self.padding)
        output += self.biases
        self.layer_output = np.maximum(0, output)
        return self.layer_output

    def calculateWeightGradients(self, dL_dY):
        dL_dW_size = np.append(self.layer_input.shape[0], self.weights.shape)
        dL_dW = np.zeros(dL_dW_size)
        dL_dY = np.squeeze(dL_dY, axis=0)

        for c in range(dL_dY.shape[2]):
            for d in range(self.layer_input.shape[3]):
                input_tensor = self.layer_input[:, :, :, d, np.newaxis]
                kernel = dL_dY[:, :, c, np.newaxis, np.newaxis]
                dL_dW[:, :, :, d, c, np.newaxis] = self.crossCorrelate(input_tensor, kernel)

        return dL_dW

    def calculateBiasGradients(self, dL_dY):
        return np.squeeze(dL_dY, axis=0)

    def calculateInputGradients(self, dL_dY):
        weights = np.swapaxes(self.weights, 2, 3)

        _, input_h, input_w, _ = self.layer_input.shape
        _, gradient_h, gradient_w, _ = dL_dY.shape

        # Assuming symmetric images/filters for now
        padding = 0
        filter_size = weights.shape[0]
        if gradient_w < input_w:
            padding = int((input_w - 1 - gradient_w + filter_size) / 2)

        return self.conv2d(dL_dY, weights, padding=padding)

    def updateWeights(self, dL_dY, learning_rate):
        output_h, output_w, num_filters = self.output_size

        if dL_dY.ndim == 1:
            dL_dY = dL_dY.reshape(1, output_h, output_w, num_filters)

        dL_dY = np.repeat(dL_dY, self.layer_output.shape[0], axis=0)
        dL_dY[np.where(self.layer_output <= 0)] = 0
        dL_dY = np.mean(dL_dY, axis=0)[np.newaxis, :]

        dL_dW = self.calculateWeightGradients(dL_dY)

        dL_dW = np.mean(dL_dW, axis=0)
        self.weights -= learning_rate * dL_dW

        dL_dB = self.calculateBiasGradients(dL_dY)
        self.biases -= learning_rate * dL_dB

        dL_dX = self.calculateInputGradients(dL_dY)
        return dL_dX

class FCLayer(Layer):
    biases = None
    weights = None
    layer_input = None

    def __init__(self, input_size, num_nodes):
        c, h, w = input_size
        limit = 1 / math.sqrt(60000)
        self.weights = np.random.uniform(low=-limit, high=limit, size=(c * h * w, num_nodes))
        self.biases = np.zeros(self.weights.shape[1])

    # Tensor is size (n, c, h, w)
    def calculateOutput(self, tensor, training):
        flattened = tensor.reshape(tensor.shape[0], -1)

        if training:
            self.layer_input = flattened

        product = np.dot(flattened, self.weights[:, :])
        product += self.biases
        max_val = np.max(product, axis=1)
        exp = np.exp(product - max_val[:, np.newaxis])
        normalized_exp = exp / np.sum(exp, axis=1)[:, np.newaxis]
        return normalized_exp

    def calculateWeightGradients(self):
        return self.layer_input

    def calculateBiasGradients(self, dL_dY):
        return np.mean(dL_dY, axis=0)

    def updateWeights(self, dL_dY, learning_rate):
        dY_dX = self.weights
        dL_dY_mean = np.mean(dL_dY, axis=0)
        dL_dX = dL_dY_mean * dY_dX
        dL_dX = np.sum(dL_dX, axis=1)

        dY_dW = self.calculateWeightGradients()
        dL_dW = np.einsum('bn,bc->bnc', dY_dW, dL_dY)
        dL_dW = np.sum(dL_dW, axis=0)

        self.weights -= learning_rate * dL_dW

        dL_dB = self.calculateBiasGradients(dL_dY)
        self.biases -= learning_rate * dL_dB

        return dL_dX

class MnistCNN:
    layers = []

    def addConvLayer(self, input_size, filter_size, num_filters, stride, padding):
        layer = ConvLayer(input_size, filter_size, num_filters, stride, padding)
        self.layers.append(layer)

        output_h, output_w, _ = layer.output_size
        return (num_filters, output_h, output_w)

    def addFCLayer(self, input_size, num_nodes):
        self.layers.append(FCLayer(input_size, num_nodes))

    def forwardPass(self, batch):
        layer_input = batch[:, :, :, np.newaxis]
        for layer in self.layers:
            output = layer.calculateOutput(layer_input, True)
            layer_input = output
        return output

    def calculateLossGradient(self, scores, labels):
        examples = labels.shape[0]
        grad = scores
        for idx, label in enumerate(labels):
            grad[idx, label] -= 1
        return grad

    def backwardPass(self, scores, labels, learning_rate):
        upstream_gradients = self.calculateLossGradient(scores, labels)
        for layer in reversed(self.layers):
            upstream_gradients = layer.updateWeights(upstream_gradients, learning_rate)

    def calculateLoss(self, scores, labels):
        examples = labels.shape[0]
        scores += 1e-15
        log_likelihood = -np.log(scores[range(examples), labels])
        loss = np.sum(log_likelihood) / examples
        return loss

    def calculateCorrectExamples(self, scores, labels):
        correct = 0
        for idx, label in enumerate(labels):
            if np.argmax(scores[idx]) == label:
                correct += 1
        return correct

    def calculateLossAndAccuracy(self, input, labels):
        scores = self.forwardPass(input)
        loss = self.calculateLoss(scores, labels)
        accuracy = 100 * self.calculateCorrectExamples(scores, labels) / labels.shape[0]

        return (loss, accuracy)

    def train(self, train_x, train_y, test_x, test_y, batch_size, epochs, learning_rate):
        test_loss, test_accuracy = self.calculateLossAndAccuracy(test_x, test_y)
        print("Initial Test Loss: %.3f  |  Initial Test Accuracy: %.3f%%\n" % (test_loss, test_accuracy))
        print("Training...")

        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0

            for x in range(0, train_x.shape[0] - batch_size, batch_size):
                scores = self.forwardPass(train_x[x : x + batch_size])

                labels = train_y[x : x + batch_size]
                loss = self.calculateLoss(scores, labels)
                total_loss += loss

                total_correct += self.calculateCorrectExamples(scores, labels)
                # print("Batch: %03d  |  Loss: %.2f" % (int(x / batch_size), loss))
                self.backwardPass(scores, labels, learning_rate)

            accuracy = 100 * total_correct / train_y.shape[0]

            test_loss, test_accuracy = self.calculateLossAndAccuracy(test_x, test_y)

            print("Epoch:", epoch + 1)
            print("   Train Loss: %.3f  |  Train Accuracy: %.3f%%" % (total_loss, accuracy), end='')
            print("  |  Test Loss:  %.3f  |  Test Accuracy:  %.3f%%" % (test_loss, test_accuracy))

    def visualize(self, examples_x, examples_y, num_examples):
        start_index = randrange(examples_y.shape[0] - num_examples)
        input_x = examples_x[start_index:start_index + num_examples]
        labels = examples_y[start_index:start_index + num_examples]
        scores = self.forwardPass(input_x)

        fig = plt.figure(figsize=(10, 7))
        fig.canvas.manager.set_window_title("Test Examples")
        rows = 2
        columns = int(num_examples / rows)
        for idx, score in enumerate(scores):
            fig.add_subplot(rows, columns, idx + 1)
            title = "Guess: %d, Actual: %d" % (np.argmax(score), labels[idx])
            plt.title(title)
            plt.axis('off')
            plt.imshow(input_x[idx], cmap='gray')
        plt.show()

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    print("Loading MNIST Dataset...")
    (train_x, train_y), (test_x, test_y) = mnist.load_data()
    # train_x = train_x[:100, :, :]
    # train_y = train_y[:100]

    num_examples, h, w = train_x.shape
    c = 1
    num_classes = 10

    cnn = MnistCNN()
    # c, h, w = cnn.addConvLayer((c, h, w), filter_size=3, num_filters=5, stride=1, padding=0)
    c, h, w = cnn.addConvLayer((c, h, w), filter_size=3, num_filters=6, stride=1, padding=0)
    cnn.addFCLayer((c, h, w), num_classes)

    cnn.train(train_x, train_y, test_x, test_y, batch_size=1000, epochs=10, learning_rate=1e-5)

    cnn.visualize(test_x, test_y, 8)
