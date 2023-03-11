import numpy as np
import newNet as nn
from newNet import Activation, NetType, Loss, Optimizer
import matplotlib.pyplot as plt

if __name__ == '__main__':

    nn.setup(
        _optimizer=Optimizer.Gradient_descent,
        _loss=Loss.Categorical_cross_entropy
    )

    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.ReLU, input_len=28*28, output_len=64)
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.ReLU, input_len=64, output_len=64)
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.SoftMax, input_len=64, output_len=10)

    # nn.a_loop(np.array([0.5, 0.15], dtype='float64'), np.array([1, 0], dtype='float64'))

    # nn.print_neuron_info()

    # inputS, outputS = [], []
    # for i in range(100):
    #     inputS.append(np.full((5,), i))
    #     outputS.append(np.full((2, ), i / 60000))
    #
    # inputS = np.array(inputS)
    # outputS = np.array(outputS)

    inputS = np.load("Digits\\l_img.npy")
    l_info = np.load("Digits\\l_info.npy")
    inputS = inputS / 255
    inputS : np.ndarray
    inputS.shape = (len(inputS), 28*28, )

    outputS = []
    for i in l_info:
        arr = np.zeros((10, ))
        arr[i] = 1
        outputS.append(arr)
    outputS = np.array(outputS, dtype='float64')

    nn.print_len = 15
    nn.train(inputS, outputS, 0.00002, 5, 32)
    nn.plot_t()


    # nn.a_loop(np.array([0.3, 0.5], dtype='float64'), np.array([1, 0], dtype='float64'))

    # nn.print_neuron_info()
    # nn.print_netInfo()
    # nn.print_weights()

    # տեղեկություն :: : : : արդեն դատասետ տերմինի փոխարեն
    # փոխանցվում է օրգինալ դատասետը՝ ինդեքսային հաջորդականության հետ միասին, այսինքն պետք է շաֆլ անել ինդեքսների
    # հաջորդականությունը