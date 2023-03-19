import time, math

import numpy as np
import matrixparallel as mx
from enum import Enum
from numba import njit
import matplotlib.pyplot as plt

class NetType(Enum):
    Perceptron = "Perceptron"
    DropOut = "DropOut"

class Activation(Enum):
    ReLU = "ReLU"
    Sigmoid = "Sigmoid"
    SoftMax = "SoftMax"
    Linear = "Linear"
    TanH = "TanH"
    NonE = "None"

class Loss(Enum):
    Categorical_cross_entropy = "Categorical Cross Entropy"
    Quadratic_loss = "Quadratic Loss"

class Optimizer(Enum):
    Gradient_descent = "Gradient Descent"
    ADAM = "ADAM"

# net setup section
weights = []
inputs = []
outputs = []

loss : Loss
optimizer : Optimizer
netInfo = []
perceptron_counter = 0
dropout_counter = 0
dropout_weights = []

error_function = []
print_len = 0

def setup(_optimizer : Optimizer, _loss : Loss):
    global loss, optimizer, inputs, outputs
    loss, optimizer = _loss, _optimizer
    inputs, outputs = [], []

def add_layer(net_type : NetType, activation : Activation, input_len : int, output_len : float):
    global inputs, outputs, perceptron_counter, dropout_counter
    if net_type == NetType.Perceptron:
        netInfo.append((net_type, activation, input_len, float(output_len), perceptron_counter))
        inputs.append(np.zeros((input_len,), dtype='float64'))
        outputs.append(np.zeros((output_len,), dtype='float64'))
        weights.append(np.random.random((input_len + 1, output_len)))
        perceptron_counter += 1
    else:
        netInfo.append((net_type, activation, input_len, output_len, dropout_counter))
        arr = np.full((int(input_len * output_len), ), 0, dtype='f8')
        arr = np.append(arr, np.full((input_len - int(input_len * output_len), ), 1, dtype='f8'))
        dropout_weights.append(np.copy(arr))
        dropout_counter += 1
    
# neural section
def forward(input_: np.ndarray) -> np.ndarray:
    data_fp = (input_, netInfo, inputs, outputs, weights, True, dropout_weights)
    forward_propagation(data_fp)
    return outputs[-1]

@njit
def forward_propagation(data: tuple):
    input_, netInfo_, inputs_, outputs_, weights_, normal, dropout_weights_ = data

    for index, info in enumerate(netInfo_):
        if info[0] == NetType.Perceptron:
            mx.jit_equal1d(inputs_[info[4]], input_)
            data_pfp = (info[1], outputs_[info[4]], inputs_[info[4]], weights_[info[4]])
            mx.forward_propagation_for_perceptron(data_pfp)
            input_ = outputs_[info[4]]
        elif normal is False and info[0] == NetType.DropOut:
            for i in range(len(dropout_weights_)):
                input_[i] *= dropout_weights_[info[4]][i]

@njit
def back_propagation(data : tuple):
    output_, netInfo_, inputs_, outputs_, weights_, gradients_, loss_, dropout_weights_ = data
    last_neuron_layer = outputs_[-1]

    mx.pre_back((loss_, netInfo_, output_, last_neuron_layer))

    for info in netInfo_[::-1]:
        if info[0] is NetType.Perceptron:
            data_bpfp = (info[1], output_, outputs_[info[4]], inputs_[info[4]], weights_[info[4]], gradients_[info[4]])
            mx.back_propagation_for_perceptron(data_bpfp)
            output_ = inputs_[info[4]]
        else:
            if info[0] == NetType.DropOut:
                for i in range(len(dropout_weights_)):
                    output_[i] *= dropout_weights_[info[4]][i]

@njit
def jit_equal1d(arr1d_1 : np.ndarray, arr1d_2 : np.ndarray, new : bool = False):
    if new is True:
        arr1d_1 = np.zeros(arr1d_2.shape, dtype=arr1d_2.dtype)
    for ind in range(len(arr1d_2)):
        arr1d_1[ind] = arr1d_2[ind]
def train(inputS : np.ndarray, outputS : np.ndarray, learning_rate : float, levels : int, batch_len : int):
    data_set_len = len(inputS)
    if batch_len > data_set_len or batch_len == 0:
        batch_len = data_set_len

    percent = int(0. * data_set_len) #* 100 / batch_len

    train_set_len = data_set_len - percent
    test_set_len = data_set_len - train_set_len
    if percent != 0:
        test_set_len += train_set_len % batch_len
        train_set_len -= train_set_len % batch_len

    train_set_index_list = np.array([i for i in range(train_set_len)], dtype='int64')
    test_set_index_list = np.array([i for i in range(train_set_len, data_set_len)], dtype='int64')

    gradient, momentum1, momentum2 = [], [], []
    for block in weights:
        gradient.append(np.zeros(block.shape, dtype='f8'))
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

    p_p = 15
    global error_function, dropout_weights
    if iterations_count % p_p != 0:
        error_function = np.zeros((2, int(iterations_count / p_p) + 1), dtype='float64')
    else:
        error_function = np.zeros((2, int(iterations_count / p_p)), dtype='float64')

    it_set : np.ndarray
    if print_len == 0 or print_len > iterations_count:
        it_set = np.array([[0, iterations_count]])
    else:
        it_1_count = iterations_count // print_len
        it_set = np.array([[i*it_1_count, (i + 1)*it_1_count] for i in range(print_len)])
        if iterations_count % print_len != 0:
            it_set[print_len - 1][1] = iterations_count

    hps = np.array([btchs, lvls, t])
    parts = np.array(parts)
    alp = np.array([alp])

    it_set[0][0] = 1
    it_set = np.append(np.array([0, 1]), it_set)
    it_set.shape = (int(len(it_set) / 2), 2)

    np.random.shuffle(train_set_index_list)

    i = 0
    for it_part in it_set:
        ts = (inputS, outputS, train_set_index_list, test_set_index_list) # problem
        # print("L : {}/{}, b {}/{} ".format(lvls + 1, levels, btchs + 1, parts_count), end='')
        if len(dropout_weights) == 0:
            dropout_weights = np.array([[1]], dtype='f8')
        whole_data = (weights, inputs, outputs, loss, optimizer, netInfo, hps, parts_count, batch_len, b1, b2, eps, alp,
                      ts, it_part, gradient, momentum1, momentum2, parts, error_function, p_p, dropout_weights)

        tme = time.time()
        train_jit(whole_data)
        btchs, lvls, t = hps[0], hps[1], hps[2]
        tme2 = time.time() - tme
        i += 1
        print("Pr: {}/{} - Iteration Number: {}/{}, Progress Time: {}s, ETA: {}".format(i, len(it_set), t,
                                                iterations_count, round(tme2, 2), convert_seconds(tme2*(print_len + 1 - i))))


@njit
def train_jit(data : tuple):
    weights_, inputs_, outputs_, loss_, optimizer_, netInfo_, hps, parts_count, batch_len, b1, b2, eps, alp, \
    ts, it_part, gradient, momentum1, momentum2, parts, error_function_, p_p, dropout_weights_ = data
    inputS, outputS, train_set_index_list, test_set_index_list = ts

    sp = alp[0]
    train_err, err_counter = 0, 0
    for ip in range(it_part[0], it_part[1]):
        # if hps[1] > 1:
        #     sp = alp[0]/ (0.001 * hps[2])
        # else:
        #     sp = alp[0]
        for i in range(len(dropout_weights_)):
            np.random.shuffle(dropout_weights_[i])
        for p in range(parts[hps[0]][0], parts[hps[0]][1] + 1):
            data_fp = (np.copy(inputS[train_set_index_list[p]]), netInfo_, inputs_, outputs_, weights_, False,
                       dropout_weights_)
            forward_propagation(data_fp)
            train_err += Loss_Calculator((np.copy(outputS[train_set_index_list[p]]), outputs_[-1], loss_))
            err_counter += 1
            data_bp = (np.copy(outputS[train_set_index_list[p]]), netInfo_, inputs_, outputs_, weights_, gradient,
                       loss_, dropout_weights_)
            back_propagation(data_bp)

        if hps[2] % p_p == 0:
            error_function_[0][int(hps[2]/p_p)] = 10 * train_err / err_counter
            train_err, err_counter = 0, 0

        for i in range(len(gradient)):
            if optimizer_ == Optimizer.Gradient_descent:
                 mx.action_by_number_jit(gradient[i], gradient[i], batch_len, "div")
                 mx.action_by_number_jit(gradient[i], gradient[i], sp, "mul")
                 mx.action_matrices_jit(weights_[i], weights_[i], gradient[i], "sub")
                 mx.action_by_number_jit(gradient[i], gradient[i], 0, "mul")
            elif optimizer_ == Optimizer.ADAM:
                ADAM((i, weights_, momentum1, momentum2, gradient, b1, b2, hps[2] + 1, eps, sp, batch_len))

        # test_error = 0
        # if len(test_set_index_list) != 0 and hps[2] % p_p == 0:
        #     for test_i in test_set_index_list:
        #         predata_i = np.copy(inputS[test_i])
        #         data_fp = (predata_i, netInfo_, inputs_, outputs_, weights_, False, dropout_weights_)
        #         forward_propagation(data_fp)
        #         predata_o = np.copy(outputS[test_i])
        #         test_error += Loss_Calculator((predata_o, outputs_[-1], loss_))
        #     error_function_[1][int(hps[2] / p_p)] = 10 * test_error / len(test_set_index_list)

        hps[2], hps[0] = hps[2] + 1, hps[0] + 1
        if hps[2] % parts_count == 0:
            np.random.shuffle(train_set_index_list)
            hps[0], hps[1] = 0, hps[1] + 1

def a_loop(input_ : np.ndarray, output_ : np.ndarray, printb : bool = False):
    global dropout_weights
    if len(dropout_weights) == 0:
        dropout_weights = np.array([[1]], dtype='f8')
    data_fp = (input_, netInfo, inputs, outputs, weights, False, dropout_weights)
    forward_propagation(data_fp)

    if printb is True:
        print_neuron_info()

    gradients = []
    for block in weights:
        gradients.append(np.zeros(block.shape, dtype='float64'))
    data_bp = (output_, netInfo, inputs, outputs, weights, gradients, loss, dropout_weights)
    back_propagation(data_bp)

    for i, block in enumerate(gradients):
        print("Gradient [{}]:".format(i))
        if printb is True:
            mx.print_matrix(block, '\t')

    print()

@njit
def ADAM(data : tuple):
    i, weights_, momentum1, momentum2, gradient, b1, b2, t, eps, alp, batch_len = data

    a1, a2, a = np.zeros(weights_[i].shape, dtype='float64'), \
                np.zeros(weights_[i].shape, dtype='float64'), \
                np.zeros(weights_[i].shape, dtype='float64')

    mx.action_by_number_jit(gradient[i], gradient[i], batch_len, "div")

    mx.action_by_number_jit(a1, momentum1[i], b1, "mul")
    mx.action_by_number_jit(a2, gradient[i], 1 - b1, "mul")
    mx.action_matrices_jit(momentum1[i], a1, a2, "add")

    mx.action_by_number_jit(a1, momentum2[i], b2, "mul")
    mx.exp_matrices_jit(a, gradient[i], 2.0)
    mx.action_by_number_jit(a2, a, 1 - b2, "mul")
    mx.action_matrices_jit(momentum2[i], a1, a2, "add")

    mx.action_by_number_jit(a1, momentum1[i], (1 - b1 ** t), "div")
    mx.action_by_number_jit(a2, momentum2[i], (1 - b2 ** t), "div")

    mx.action_by_number_jit(a1, a1, alp, "mul")
    mx.exp_matrices_jit(a2, a2, 0.5)
    mx.action_by_number_jit(a2, a2, eps, "add")
    mx.action_matrices_jit(a, a1, a2, "div")
    mx.action_matrices_jit(weights_[i], weights_[i], a, "sub")

    mx.action_by_number_jit(gradient[i], gradient[i], 0, "mul")

@njit
def Loss_Calculator(data : tuple):
    output_data, last_layer_output, loss_ = data
    if loss_ == Loss.Categorical_cross_entropy:
        return Cross_Entropy((output_data, last_layer_output))
    elif loss_ == Loss.Quadratic_loss:
        return Quadratic_loss((output_data, last_layer_output))

@njit
def Cross_Entropy(data : tuple):
    output, last_layer_output = data
    E = 0
    for a_i, a_value in enumerate(last_layer_output):
        if output[a_i] != 0:
            E += output[a_i] * np.log(a_value)
    return -E

@njit
def Quadratic_loss(data : tuple):
    output, last_layer_output = data
    E = 0
    for a_i, a_value in enumerate(last_layer_output):
        E += (output[a_i] - a_value) ** 2
    return E

# additional functions
def convert_seconds(seconds):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, remaining_seconds)
def plot_t():
    plt.plot(error_function[0])
    plt.plot(error_function[1])
    plt.ylabel('Cost')
    plt.xlabel('Times')
    plt.show()

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
def cut_trail(f_str):
    cut = 0
    for c in f_str[::-1]:
        if c == "0":
            cut += 1
        else:
            break
    if cut == 0:
        for c in f_str[::-1]:
            if c == "9":
                cut += 1
            else:
                cut -= 1
                break
    if cut > 0:
        f_str = f_str[:-cut]
    if f_str == "":
        f_str = "0"
    return f_str

@njit
def float2str(value):
    if math.isnan(value):
        return "nan"
    elif value == 0.0:
        return "0.0"
    elif value < 0.0:
        return "-" + float2str(-value)
    elif math.isinf(value):
        return "inf"
    else:
        max_digits = 16
        min_digits = -4
        e10 = math.floor(math.log10(value)) if value != 0.0 else 0
        if min_digits < e10 < max_digits:
            i_part = math.floor(value)
            f_part = math.floor((1 + value % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits - e10])
            return i_str + "." + f_str
        else:
            m10 = value / 10.0 ** e10
            i_part = math.floor(m10)
            f_part = math.floor((1 + m10 % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits])
            e_str = str(e10)
            if e10 >= 0:
                e_str = "+" + e_str
            return i_str + "." + f_str + "e" + e_str

# print section
def save():
    for i in range(len(weights)):
        np.save(f"weight{i}.npy", weights[i])

def load():
    for i in range(len(weights)):
        weights[i] = np.load(f"Digits\\weight{i}.npy")
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


def randomF():
    for i in range(len(weights)):
        print(np.sum(weights[i]))
