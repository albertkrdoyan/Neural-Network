import numpy as np
import newNet as nn
from newNet import Activation, NetType, Loss, Optimizer
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nn.setup(
        _optimizer=Optimizer.Gradient_descent,
        _loss=Loss.Categorical_cross_entropy
    )

    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.ReLU, input_len=28*28, output_len=128)
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.SoftMax, input_len=128, output_len=10)

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

    # print(nn.weights[0].shape)
    # print(nn.weights[1].shape)
    # nn.load()
    # print(nn.weights[0].shape)
    # print(nn.weights[1].shape)

    # inputS = inputS[0 : 6000]
    # outputS = outputS[0 : 6000]

    nn.print_len = 15
    nn.train(inputS, outputS, 0.025, 3, 32)
    nn.plot_t()
    nn.save()

    t_img = np.load("Digits\\t_img.npy")
    t_info = np.load("Digits\\t_info.npy")
    t_img : np.ndarray
    t_img = t_img / 255
    t_img.shape = (len(t_img), 28*28)

    print("Calculating...")
    bad_answers = []
    rights = 0
    for i in range(len(t_img)):
        data = nn.argmax(nn.forward(t_img[i]))

        if data[t_info[i]] == 1:
            rights += 1
        else:
            bad_answers.append(t_img[i])

    print("Correct: {}%".format(rights / 100))
    print("END")

    # bad_answers = np.array(bad_answers)
    # bad_answers.shape = (len(bad_answers), 28, 28)
    # rng = 100 if len(bad_answers) > 100 else len(bad_answers)
    # for i in range(rng):
    #     nn.plt.subplot(10, 10, i + 1)
    #     nn.plt.xticks([])
    #     nn.plt.yticks([])
    #     nn.plt.imshow(bad_answers[i], cmap=nn.plt.cm.binary)
    #
    # nn.plt.show()
