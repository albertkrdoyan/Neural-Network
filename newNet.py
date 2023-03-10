import time

import numpy as np
import matrix as mx
from enum import Enum
from numba import njit

class NetType(Enum):
    Perceptron = "Perceptron"

class Activation(Enum):
    ReLU = "ReLU"
    Sigmoid = "Sigmoid"
    SoftMax = "SoftMax"
    Linear = "Linear"
    TanH = "TanH"

class Loss(Enum):
    Categorical_cross_entropy = "Categorical Cross Entropy"
    Quadratic_loss = "Quadratic Loss"

class Optimizer(Enum):
    Gradient_descent = "Gradient Descent"
    Adam = "Adam"

# net setup section
weights = []
inputs = []
outputs = []

loss : Loss
optimizer : Optimizer
netInfo = []
perceptron_counter = 0

print_len = 0

def setup(_optimizer : Optimizer, _loss : Loss):
    global loss, optimizer, inputs, outputs
    loss, optimizer = _loss, _optimizer
    inputs, outputs = [], []

def add_layer(net_type : NetType, activation : Activation, input_len : int, output_len : int):
    global inputs, outputs, perceptron_counter
    if net_type == NetType.Perceptron:
        netInfo.append((net_type, activation, input_len, output_len, perceptron_counter))
        inputs.append(np.zeros((input_len,), dtype='float64'))
        outputs.append(np.zeros((output_len,), dtype='float64'))
        weights.append(np.random.random((input_len + 1, output_len)))
        perceptron_counter += 1
    else:
        #Dropout, ect...
        pass

# neural section
def forward(input_: np.ndarray) -> object:
    data_fp = (input_, netInfo, inputs, outputs, weights, True)
    forward_propagation(data_fp)
    return outputs[-1]

@njit(debug=True)
def forward_propagation(data: tuple):
    input_, netInfo_, inputs_, outputs_, weights_, normal = data

    for index, info in enumerate(netInfo_):
        if info[0] == NetType.Perceptron:
            jit_equal1d(inputs_[index], input_)
            data_pfp = (info[1], outputs_[index], inputs_[index], weights_[index])
            forward_propagation_for_perceptron(data_pfp)
            input_ = outputs_[index]
        elif normal is False:
            #Dropout, CNN ect.
            pass

@njit
def forward_propagation_for_perceptron(data : tuple):
    activation, outputs_, inputs_, weights_ = data
    mx.neuralMultiplicationMega(outputs_, inputs_, weights_)

    if activation == Activation.ReLU:
        for i in range(len(outputs_)):
            if outputs_[i] < 0:
                outputs_[i] = 0
    elif activation == Activation.Sigmoid:
        for i in range(len(outputs_)):
            outputs_[i] = 1/(1 + np.exp(-outputs_[i]))
    elif activation == Activation.SoftMax:
        outputs_ = mx.SoftMax(outputs_)
    elif activation == Activation.TanH:
        for i in range(len(outputs_)):
            outputs_[i] = np.tanh(-outputs_[i])

@njit(debug=True)
def back_propagation(data : tuple):
    output_, netInfo_, inputs_, outputs_, weights_, gradients_, loss_ = data

    last_neuron_layer = outputs_[-1]

    if loss_ == Loss.Categorical_cross_entropy:
        if netInfo_[-1][1] == Activation.SoftMax:
            for i in range(len(last_neuron_layer)):
                output_[i] = last_neuron_layer[i] - output_[i]
        else:
            for i in range(len(last_neuron_layer)):
                output_[i] = - output_[i] / last_neuron_layer[i]
    elif loss_ == Loss.Quadratic_loss:
        for i in range(len(last_neuron_layer)):
            output_[i] = 2 * (last_neuron_layer[i] - output_[i])

    netInfo__ = []
    for i in range(len(netInfo_) - 1, -1, -1):
        netInfo__.append(netInfo_[i])

    for info in netInfo__:
        if info[0] is NetType.Perceptron:
            data_bpfp = (info[1], output_, outputs_[info[4]], inputs_[info[4]], weights_[info[4]], gradients_[info[4]])
            back_propagation_for_perceptron(data_bpfp)
            output_ = inputs_[info[4]]
        else:
            #DropOut
            pass

@njit
def back_propagation_for_perceptron(data : tuple):
    activation, output_, outputs_, inputs_, weights_, gradient_ = data

    if activation == Activation.ReLU:
        for i in range(len(outputs_)):
            if outputs_[i] == 0:
                output_[i] = 0
    elif activation == Activation.Sigmoid:
        for i in range(len(outputs_)):
            output_[i] = output_[i] * outputs_[i] * (1 - outputs_[i])
    elif activation == Activation.TanH:
        for i in range(len(outputs_)):
            output_[i] = output_[i] * (1 - outputs_[i]**2)

    derivative = np.zeros_like(inputs_, dtype='float64')
    for i in range(len(derivative)):
        for j in range(len(output_)):
            derivative[i] += output_[j] * weights_[i][j]

    bias_line = len(gradient_) - 1
    for i, val in enumerate(output_):
        gradient_[bias_line][i] += val
        for j, val_j in enumerate(inputs_):
            gradient_[j][i] += (val * val_j)

    jit_equal1d(inputs_, derivative)

def train(inputS : np.ndarray, outputS : np.ndarray, learning_rate : float, levels : int, batch_len : int):
    data_set_len = len(inputS)
    if batch_len > data_set_len or batch_len == 0:
        batch_len = data_set_len

    percent = 0.2 * 100 / batch_len

    train_set_len = data_set_len - int(data_set_len * percent / 100)
    test_set_len = data_set_len - train_set_len
    test_set_len += train_set_len % batch_len
    train_set_len -= train_set_len % batch_len

    train_set_index_list = [i for i in range(train_set_len)]
    test_set_index_list = [i for i in range(train_set_len, data_set_len)]

    gradient, momentum1, momentum2 = [], [], []
    for block in weights:
        gradient.append(np.zeros(block.shape, dtype='f8'))
        if optimizer == Optimizer.Adam:
            momentum1.append(np.zeros(block.shape, dtype='f8'))
            momentum2.append(np.zeros(block.shape, dtype='f8'))

    b1, b2, eps, = 0.9, 0.999, 1e-7
    alp, t = float(learning_rate), 0

    parts_count = train_set_len // batch_len
    parts = [[i * batch_len, (i + 1) * batch_len - 1] for i in range(parts_count)]
    if train_set_len % batch_len != 0:
        parts.append([parts_count * batch_len, train_set_len - 1])
    parts_count = len(parts) #

    iterations_count = levels * parts_count

    print("Length of all data set : {}".format(data_set_len))
    print("Length of train set : {}".format(train_set_len))
    print("Length of test set : {}".format(test_set_len))
    print("Batch count: {}, length of a batch: {}".format(parts_count, batch_len))
    print("Training levels count: {}".format(levels))
    print("All iterations count: {}\n".format(iterations_count))

    lvls = 0
    btchs = 0

    it_set : list
    if print_len == 0 or print_len > iterations_count:
        it_set = [[0, iterations_count]]
    else:
        it_1_count = iterations_count // print_len
        it_set = [[i*it_1_count, (i + 1)*it_1_count] for i in range(print_len)]
        if iterations_count % print_len != 0:
            it_set.append([print_len*it_1_count, iterations_count])

    hps = np.array([btchs, lvls, t])
    parts = np.array(parts)

    for it_part in it_set:
        ts = (inputS, outputS, train_set_index_list, test_set_index_list) # problem
        # print("L : {}/{}, b {}/{} ".format(lvls + 1, levels, btchs + 1, parts_count), end='')
        whole_data = (weights, inputs, outputs, loss, optimizer, netInfo, hps, parts_count, b1, b2, eps, alp,
                      ts, it_part, gradient, momentum1, momentum2, parts)

        tme = time.time()
        train_jit(whole_data)
        btchs, lvls, t = hps[0], hps[1], hps[2]
        tme2 = time.time() - tme
        print("Iteration Number: {}/{}, Time: {}s.".format(t, iterations_count, round(tme2, 2)))


@njit
def train_jit(data : tuple):
    # weights_, inputs_, outputs_, loss_, optimizer_, netInfo_, hps, parts_count, b1, b2, eps, alp,\
    # train_set, test_set, it_part, matrices, hyper_parameters, parts, lvl_btch = data
    #
    # for ip in range(it_part[0], it_part[1]):
    #     for p in range(parts[hps[0]][0], parts[hps[0]][1]):
    #         pass
    #
    #     hps[2] += 1
    #     hps[0] += 1
    #     if hps[2] % parts_count == 0:
    #         hps[0] = 0
    #         hps[1] += 1
    pass

def a_loop(input_ : np.ndarray, output_ : np.ndarray):
    data_fp = (input_, netInfo, inputs, outputs, weights, False)
    forward_propagation(data_fp)

    gradients = []
    for block in weights:
        gradients.append(np.zeros(block.shape, dtype='float64'))
    data_bp = (output_, netInfo, inputs, outputs, weights, gradients, loss)
    back_propagation(data_bp)

    for i, block in enumerate(gradients):
        print("Gradient [{}]:".format(i))
        mx.print_matrix(block, '\t')

    print()

# additional functions
def argmax(arr_: np.ndarray):
    result_ = [0. for _ in range(len(arr_))]
    max_ = arr_[0]
    max_index = 0

    for ind in range(1, len(arr_)):
        if arr_[ind] > max_:
            max_ = arr_[ind]
            max_index = ind

    for ind in range(len(result_)):
        if ind == max_index:
            result_[ind] = 1
            return result_

@njit
def jit_equal1d(arr1d_1 : np.ndarray, arr1d_2 : np.ndarray):
    for ind, el in enumerate(arr1d_2):
        arr1d_1[ind] = el

# print section
def print_neuron_info():
    for i in range(len(inputs)):
        if i == 0:
            print("Neuron layer [{}]: ".format(i), end='')
            print(inputs[i])
        print("Neuron layer [{}]: ".format(i + 1), end='')
        print(outputs[i])
    print()

def print_netInfo():
    print("Optimization method: \"{}\"\nLoss function: \"{}\"\n".format(optimizer, loss))
    for i in range(len(netInfo)):
        print("Net: [{}]:\n\t Type: {}.\n\t Activation function: {}.\n\t IN: {}, OUT: {}".format(
            i, netInfo[i][0].value, netInfo[i][1].value, netInfo[i][2], netInfo[i][3]))
    print()

def print_weights():
    for i in range(len(weights)):
        print("Weight [{}]".format(i))
        mx.print_matrix(weights[i], '\t')