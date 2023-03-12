import numpy as np
import newNet as nn
from newNet import Activation, NetType, Loss, Optimizer

if __name__ == '__main__':
    nn.setup(
        _optimizer=Optimizer.Gradient_descent,
        _loss=Loss.Categorical_cross_entropy
    )
    
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.ReLU, input_len=2, output_len=5)
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.SoftMax, input_len=5, output_len=2)

    inputs, outputs = [], []
    len_ = 10000
    for i in range(len_):
        inputs.append(np.array([i/len_, 1], dtype='f8'))
        if i < 7 * len_/10:
            outputs.append(np.array([1, 0], dtype='f8'))
        else:
            outputs.append(np.array([0, 1], dtype='f8'))

    inputs, outputs = np.array(inputs), np.array(outputs)

    nn.print_len = 1

    nn.train(inputs, outputs, 0.003, 300, 32)
    nn.plot_t()

    # exit(0)

    f = [[], []]

    for i in range(len_):
        data = nn.forward(np.array([i/len_, 1], dtype='f8'))
        for fi in range(len(f)):
            f[fi].append(data[fi])

    for fi in range(len(f)):
        nn.plt.plot(f[fi])
    nn.plt.show()
    # նախորդ խնդրի պատճառը օպտիմիզացիայի ընտրությունն էր, այսինքն պետք է օգտագործել ADAM
    # լուծել վերևի խնդիրը