import numpy as np

ZERO = np.array([ 
    [0,0,0,1,0,0,0],
    [0,0,1,0,1,0,0],
    [0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],
    [0,1,0,0,0,1,0],
    [0,0,1,0,1,0,0],
    [0,0,0,1,0,0,0],
]).flatten()
ONE = np.array([
    [0,0,0,1,0,0,0],
    [0,0,1,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0],
    [0,0,1,1,1,0,0],
]).flatten()
ZERO_Y = (1,0)
ONE_Y = (0,1)

class Layer:
    def __init__(self,input_size,hidden_size,batch_size):
        self.weights = np.random.rand(hidden_size, input_size)
        self.biases = np.zeros((hidden_size, batch_size))

    def forward(self,inputs):
        return np.matmul(self.weights,inputs) + self.biases

    def backward(self, previous_inputs, output_grad, learning_rate):
        self.weights -= learning_rate * np.matmul(output_grad, previous_inputs.T)
        self.biases -= learning_rate * output_grad 
        return np.matmul(self.weights.T, output_grad)

relu = lambda x: np.maximum(0, x)
relu_derivative = lambda x: np.where(x > 0, 1, 0)
softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)

if __name__ == '__main__':
    inputs = np.array([ZERO,ONE,ZERO,ONE,ZERO]).T
    y = np.array([ZERO_Y,ONE_Y,ZERO_Y,ONE_Y,ZERO_Y]).T
    hidden_layer = Layer(inputs.shape[0],10,inputs.shape[1])
    output_layer = Layer(hidden_layer.weights.shape[0],2,inputs.shape[1])

    for epoch in range(1000):
        hidden_output = hidden_layer.forward(inputs)
        activation_output = relu(hidden_output)
        output = output_layer.forward(activation_output)
        predictions = softmax(output)
        loss = predictions - y
        indirect_loss = output_layer.backward(activation_output, loss, 0.01)
        hidden_output_grad = indirect_loss * relu_derivative(hidden_output)
        hidden_layer.backward(inputs, hidden_output_grad, 0.01)

    predict = lambda x: softmax(output_layer.forward(relu(hidden_layer.forward(x))))
    print(np.around(predict(inputs),decimals=3))
