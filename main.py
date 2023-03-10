import numpy as np
import newNet as nn
from newNet import Activation, NetType, Loss, Optimizer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nn.setup(
        _optimizer=Optimizer.Adam,
        _loss=Loss.Categorical_cross_entropy
    )

    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.ReLU, input_len=2, output_len=5)
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.ReLU, input_len=5, output_len=5)
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.SoftMax, input_len=5, output_len=2)

    inputS, outputS = [], []
    for i in range(60000):
        inputS.append(np.full((5,), i))
        outputS.append(np.full((2, ), i / 60000))

    inputS = np.array(inputS)
    outputS = np.array(outputS)

    nn.print_len = 10

    nn.train(inputS, outputS, 0.1, 5, 32)


    # nn.a_loop(np.array([0.3, 0.5], dtype='float64'), np.array([1, 0], dtype='float64'))

    # nn.print_neuron_info()
    # nn.print_netInfo()
    # nn.print_weights()
