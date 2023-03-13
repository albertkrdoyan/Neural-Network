import numpy as np
import newNet as nn
from newNet import Activation, NetType, Loss, Optimizer

if __name__ == '__main__':
    nn.setup(
        _optimizer=Optimizer.ADAM,
        _loss=Loss.Categorical_cross_entropy
    )

    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.ReLU, input_len=784, output_len=256)
    nn.add_layer(net_type=NetType.Perceptron, activation=Activation.SoftMax, input_len=256, output_len=10)

    inputS = np.load("Digits\\l_img.npy")
    l_info = np.load("Digits\\l_info.npy")
    inputS = inputS / 255
    inputS.shape = (len(inputS), 784,)

    outputS = []
    for i in l_info:
        arr = np.zeros((10, ))
        arr[i] = 1
        outputS.append(arr)
    outputS = np.array(outputS, dtype='float64')

    nn.print_len = 30
    nn.train(inputS, outputS, 0.01, 5, 32)
    nn.plot_t()

    t_img = np.load("Digits\\t_img.npy")
    t_info = np.load("Digits\\t_info.npy")
    t_img = t_img / 255
    t_img.shape = (len(t_img), 28 * 28)

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

    bad_answers = np.array(bad_answers)
    bad_answers.shape = (len(bad_answers), 28, 28)
    rng = 100 if len(bad_answers) > 100 else len(bad_answers)
    for i in range(rng):
        nn.plt.subplot(10, 10, i + 1)
        nn.plt.xticks([])
        nn.plt.yticks([])
        nn.plt.imshow(bad_answers[i], cmap=nn.plt.cm.binary)

    nn.plt.show()
