import numpy as np
import image_data as image


class NeuralNetwork():
    def init(self):
        np.random.seed(1)
        w1 = .1 * np.random.randn(784, 20)
        w2 = .1 * np.random.randn(20, 20)
        w3 = .1 * np.random.randn(20, 10)
        b1 = np.zeros((1, 20))
        b2 = np.zeros((1, 20))
        b3 = np.zeros((1, 10))
        return w1, b1, w2, b2, w3, b3

    def train(self, inputs, w1, b1, w2, b2, w3, b3, running):
        if running == True:
            a1 = np.dot(inputs, w1) + b1
            z1 = ReLu.reLu(a1)
            a2 = np.dot(w2, z1.T) + b2.T
            z2 = ReLu.reLu(a2)
            a3 = np.dot(z2.T, w3) + b3
            z3 = sftMax.sftmax(a3)
            return z3, z2, z1, a1, a2, a3

    def think(self, inputs, wnum, bnum):
        output = np.dot(inputs, wnum)
        return output

    def reLuDeriv(self, z):
        return z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y
        return one_hot_Y.T

    def backprop(self, w1, a1, z1, w2, a2, z2, a3, z3, w3, X, Y):
        m = Y.size
        one_hot_y = self.one_hot(Y)
        dz3 = a3.T - one_hot_y
        dw3 = 1 / m * dz3.dot(a2.T)
        db3 = 1 / m * np.sum(dz3)
        dz2 = w3.dot(dz3) * self.reLuDeriv(z2)
        dw2 = 1 / m * dz2.dot(a1)
        db2 = 1 / m * np.sum(dz2)
        dz1 = w2.dot(dz2).T * self.reLuDeriv(z1)
        dw1 = 1 / m * dz1.T.dot(X)
        db1 = 1 / m * np.sum(dz1)
        return dw1, db1, dw2, db2, dw3, db3

    def get_predictions(self, z3):
        predictions = []
        for i in z3:

            x = (np.argmax(i))
            predictions.append(x)
        return np.array(predictions)
    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / Y.size

    def update_params(self, w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha):
        w1 = w1 - alpha * dw1.T
        b1 = b1 - alpha * db1
        w2 = w2 - alpha * dw2.T
        b2 = b2 - alpha * db2
        w3 = w3 - alpha * dw3.T
        b3 = b3 - alpha * db3

        return w1, b1, w2, b2, w3, b3

    def graident(self, X, Y, iterations, alpha):
        w1, b1, w2, b2, w3, b3 = self.init()
        for i in range(iterations):
            z3, z2, z1, a1, a2, a3 = self.train(X, w1, b1, w2, b2, w3, b3, True)
            dw1, db1, dw2, db2, dw3, db3 = self.backprop(w1, z1, a1, w2, z2, a2, z3, a3, w3, X, Y)
            w1, b1, w2, b2, w3, b3 = self.update_params(w1, b1, w2, b2, w3, b3, dw1, db1, dw2, db2, dw3, db3, alpha)

            if i % 500 == 0:
                print("Iteration ", i)
                predictions = self.get_predictions(z3)
                print("Accuracy")
                accuracy = self.get_accuracy(predictions, Y)
                print(accuracy)
        return w1, b1, w2, b2, w3, b3


class ReLu:
    @staticmethod
    def reLu(x):
        return np.maximum(x, 0)

    @staticmethod
    def Relu_derivative(x):
        return x > 0


class sftMax:
  def sftmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)



if __name__ == "__main__":
    target_output = image. getImageData()[1]
    training_input = image.getImageData()[0]
    nn = NeuralNetwork()
    w1, b1, w2, b2, w3, b3 = nn.graident(training_input,
                                         target_output, 50000, 0.01)

