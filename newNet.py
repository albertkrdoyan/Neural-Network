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
        pass
        #Dropout, ect...

# neural section
def forward(input_ : np.ndarray):
    data_fp = (netInfo, inputs, outputs, weights)
    forward_propagation(input_, data_fp, True)
    return outputs[-1]

@njit(debug=True)
def forward_propagation(input_ : np.ndarray, data: tuple, normal = False):
    netInfo_, inputs_, outputs_, weights_ = data

    for index, info in enumerate(netInfo_):
        if info[0] == NetType.Perceptron:
            jit_equal1d(inputs_[index], input_)
            data_pfp = (outputs_[index], inputs_[index], weights_[index])
            forward_propagation_for_perceptron(info[1], data_pfp)
            input_ = outputs_[index]
        elif normal is False:
            pass
            #Dropout, CNN ect.

@njit
def forward_propagation_for_perceptron(activation : Activation, data : tuple):
    outputs_, inputs_, weights_ = data
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
def back_propagation(output_ : np.ndarray, data : tuple):
    netInfo_, inputs_, outputs_, weights_, gradients_, loss_ = data

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
            data_bpfp = (outputs_[info[4]], inputs_[info[4]], weights_[info[4]], gradients_[info[4]])
            back_propagation_for_perceptron(info[1], output_, data_bpfp)
            output_ = inputs_[info[4]]
        else:
            pass
            #DropOut

@njit
def back_propagation_for_perceptron(activation : Activation, output_ : np.ndarray, data : tuple):
    outputs_, inputs_, weights_, gradient_ = data

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
    data_set_len = len(inputS) # ստանում ենք ամբողջ տվյալների երկարությունը
    if batch_len > data_set_len or batch_len == 0:
        batch_len = data_set_len # հաստատում ենք բատչի երկարությունը

    percent = 0.2 * 100 / batch_len # որոշում է թեսթային տվյալների տոկոսային երկարությունը

    train_set_len = data_set_len - int(data_set_len * percent / 100) # ուսուցանվող տվյալների երկարությունը
    test_set_len = data_set_len - train_set_len # թեսթային տվյալների երկարությունը
    test_set_len += train_set_len % batch_len
    train_set_len -= train_set_len % batch_len

    train_set = [inputS[0 : train_set_len], outputS[0 : train_set_len]] # ուսուցանվող տվյալներ
    test_set = [inputS[train_set_len : data_set_len], outputS[train_set_len : data_set_len]] # թեսթային տվյալների

    gradient, momentum1, momentum2 = [], [], []
    for block in weights:
        gradient.append(np.zeros(block.shape, dtype='f8')) # ստեղծում է գռադիենտի մատրիցը
        if optimizer == Optimizer.Adam:
            momentum1.append(np.zeros(block.shape, dtype='f8'))
            momentum2.append(np.zeros(block.shape, dtype='f8')) # ստեղծում է 2 իմպուլսային մատրիցաներ, եթե ADAM
                                                                # օպտիմիզացիա է
    matrices = (gradient, momentum1, momentum2)

    b1, b2, eps, = 0.9, 0.999, 1e-7
    alp, t = float(learning_rate), 0 # ստեղծում է հիպերպարամետրերը

    hyper_parameters = (b1, b2, eps, alp, t)

    parts_count = train_set_len // batch_len
    parts = [[i * batch_len, (i + 1) * batch_len - 1] for i in range(parts_count)]
    if train_set_len % batch_len != 0:
        parts.append([parts_count * batch_len, train_set_len - 1])
    parts_count = len(parts) #

    iterations_count = levels * parts_count

    # print("Length of all data set : {}".format(data_set_len))
    # print("Length of train set : {}".format(train_set_len))
    # print("Length of test set : {}".format(test_set_len))
    # print("Batch count: {}, length of a batch: {}".format(parts_count, batch_len))
    # print("Training levels count: {}".format(levels))
    # print("All iterations count: {}\n".format(iterations_count))

    lvls = 0
    btchs = 0

    lvl_btch = (lvls, btchs, parts_count)

    it_set = []
    if print_len == 0:
        it_set = [[0, iterations_count]]
    else:
        it_1_count = iterations_count // print_len
        it_set = [[i*it_1_count, (i + 1)*it_1_count] for i in range(print_len)]
        if iterations_count % print_len != 0:
            it_set.append([print_len*it_1_count, iterations_count])

    data = (weights, inputs, outputs, loss, optimizer, netInfo)

    for it_part in it_set:
        lvl_btch = (lvls, btchs, parts_count)
        hyper_parameters = (b1, b2, eps, alp, t)
        data_tj = (train_set, test_set, it_part, t, matrices, hyper_parameters, parts, lvl_btch)

        tme = time.time()
        lvls, btchs, t = train_jit(data_tj, data)
        tme2 = time.time() - tme
        print("Iteration Number: {}/{}, Time: {}s.".format(t, iterations_count, round(tme2, 2)))


#@njit
def train_jit(data1 : tuple, data2 : tuple):
    train_set, test_set, it_part, t, matrices, hyper_parameters, parts, lvl_btch = data1
    weights_, inputs_, outputs_, loss_, optimizer_, netInfo_ = data2

    gradient, momentum1, momentum2 = matrices
    b1, b2, eps, alp, t = hyper_parameters
    lvls, btchs, parts_count = lvl_btch

    for ip in range(it_part[0], it_part[1]):
        for p in range(parts[btchs][0], parts[btchs][1]):
            pass
        pass
        t += 1
        btchs += 1
        if t % parts_count == 0:
            btchs = 0
            lvls += 1

    return lvls, btchs, t

def a_loop(input_, output_):
    data_fp = (netInfo, inputs, outputs, weights)
    forward_propagation(input_, data_fp)

    gradients = []
    for block in weights:
        gradients.append(np.zeros(block.shape, dtype='float64'))
    data_bp = (netInfo, inputs, outputs, weights, gradients, loss)
    back_propagation(output_, data_bp)

    for i, block in enumerate(gradients):
        print("Gradient [{}]:".format(i))
        mx.print_matrix(block, '\t')

    print()

# additional functions
def argmax(arr_: np.ndarray):
    result_ = [0 for _ in range(len(arr_))]
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