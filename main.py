import numpy as np
import newNet as nn
from newNet import Activation, NetType, Loss, Optimizer
import time

if __name__ == '__main__':

    nn.setup(
        _optimizer=Optimizer.Gradient_descent,
        _loss=Loss.Categorical_cross_entropy
    )

    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.ReLU, input_len=2, output_len=7)
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.SoftMax, input_len=7, output_len=4)

    _len = 10000
    inputs, outputs = [np.array([], dtype='float64') for _ in range(_len)], [np.array([], dtype='float64') for _ in
                                                                             range(_len)]

    for i in range(_len):
        inputs[i] = np.array([i / _len, 1])
        if _len / 10 < i < 3 * _len / 10:
            outputs[i] = np.array([1, 0, 0, 0])
        elif 4 * _len / 10 < i < 6 * _len / 10:
            outputs[i] = np.array([0, 1, 0, 0])
        elif 7 * _len / 10 < i < 9 * _len / 10:
            outputs[i] = np.array([0, 0, 1, 0])
        else:
            outputs[i] = np.array([0, 0, 0, 1])
    inputs, outputs = np.array(inputs), np.array(outputs, dtype='float64')

    print("Start training.")
    t = time.time()
    nn.train(inputs, outputs, 0.003, 1000, 32)
    print("End of training.")
    print("Duration: {}\n".format(time.time() - t))

    nn.plot_t()

    res_f, c = [[], [], [], []], len(outputs[0])
    r = 10000
    for i in range(r):
        data = nn.forward(np.array([i / r, 1]))
        for j in range(c):
            res_f[j].append(data[j])

        if i % (r / 10) == 0:
            print("Core : {}, nn : {}".format(i / r, nn.argmax(data)))

    for j in range(c):
        nn.plt.plot(res_f[j], label='co{}'.format(j))
    nn.plt.legend()
    nn.plt.show()

    # 2 ներքին շարքի դեպքում չի աշխատում.....
