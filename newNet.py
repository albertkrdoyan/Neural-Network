import time, math

import numpy as np
import matrix as mx
from enum import Enum

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
    global inputs, outputs, perceptron_counter
    if net_type == NetType.Perceptron:
        netInfo.append((net_type, activation, input_len, int(output_len), perceptron_counter))
        inputs.append(np.zeros((input_len,), dtype='float64'))
        outputs.append(np.zeros((int(output_len),), dtype='float64'))
        weights.append(np.random.random((input_len + 1, int(output_len))))
        perceptron_counter += 1
    else:
        netInfo.append((net_type, activation, input_len, output_len, dropout_counter))
        arr = np.full((int(input_len * int(output_len)), ), 0, dtype='f8')
        arr = np.append(arr, np.full((input_len - int(input_len * output_len), ), 1, dtype='f8'))
        np.random.shuffle(arr)
        dropout_weights.append(np.copy(arr))

# neural section
def forward(input_: np.ndarray) -> np.ndarray:
    data_fp = (input_, netInfo, inputs, outputs, weights, True)
    forward_propagation(data_fp)
    return outputs[-1]

#@njit(debug=True)
def forward_propagation(data: tuple):
    input_, netInfo_, inputs_, outputs_, weights_, normal = data

    for index, info in enumerate(netInfo_):
        if info[0] == NetType.Perceptron:
            jit_equal1d(inputs_[info[4]], input_)
            data_pfp = (info[1], outputs_[info[4]], inputs_[info[4]], weights_[info[4]])
            forward_propagation_for_perceptron(data_pfp)
            input_ = outputs_[info[4]]
        elif normal is False:
            
            pass

#@njit
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
        mx.SoftMax(outputs_)
    elif activation == Activation.TanH:
        for i in range(len(outputs_)):
            outputs_[i] = np.tanh(-outputs_[i])


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

    percent = 0. * 100 / batch_len

    train_set_len = data_set_len - int(data_set_len * percent / 100)
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
    global error_function
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

    it_set[0][0] = 1
    it_set = np.append(np.array([0, 1]), it_set)
    it_set.shape = (int(len(it_set) / 2), 2)

    np.random.shuffle(train_set_index_list)

    i = 0
    for it_part in it_set:
        ts = (inputS, outputS, train_set_index_list, test_set_index_list) # problem
        # print("L : {}/{}, b {}/{} ".format(lvls + 1, levels, btchs + 1, parts_count), end='')
        whole_data = (weights, inputs, outputs, loss, optimizer, netInfo, hps, parts_count, batch_len, b1, b2, eps, alp,
                      ts, it_part, gradient, momentum1, momentum2, parts, error_function, p_p)

        tme = time.time()
        train_jit(whole_data)
        btchs, lvls, t = hps[0], hps[1], hps[2]
        tme2 = time.time() - tme
        i += 1
        print("Pr: {}/{} - Iteration Number: {}/{}, Time: {}s.".format(i, len(it_set), t, iterations_count, round(tme2, 2)))


def train_jit(data : tuple):
    weights_, inputs_, outputs_, loss_, optimizer_, netInfo_, hps, parts_count, batch_len, b1, b2, eps, alp, \
    ts, it_part, gradient, momentum1, momentum2, parts, error_function_, p_p = data
    inputS, outputS, train_set_index_list, test_set_index_list = ts
    
    train_err, err_counter = 0, 0
    for ip in range(it_part[0], it_part[1]):
        for p in range(parts[hps[0]][0], parts[hps[0]][1] + 1):
            data_fp = (np.copy(inputS[train_set_index_list[p]]), netInfo_, inputs_, outputs_, weights_, False)
            forward_propagation(data_fp)
            train_err += Loss_Calculator((np.copy(outputS[train_set_index_list[p]]), outputs_[-1], loss_))
            err_counter += 1
            data_bp = (np.copy(outputS[train_set_index_list[p]]), netInfo_, inputs_, outputs_, weights_, gradient, loss_)
            back_propagation(data_bp)

        if hps[2] % p_p == 0:
            error_function_[0][int(hps[2]/p_p)] = train_err / err_counter
            train_err, err_counter = 0, 0

        for i in range(len(gradient)):
            mx.action_by_number_jit(gradient[i], gradient[i], batch_len, "div")
            if optimizer_ == Optimizer.Gradient_descent:
                 mx.action_by_number_jit(gradient[i], gradient[i], alp, "mul")
                 mx.action_matrices_jit(weights_[i], weights_[i], gradient[i], "sub")
            elif optimizer_ == Optimizer.ADAM:
                ADAM((i, weights_, momentum1, momentum2, gradient, b1, b2, hps[2] + 1, eps, alp))
            mx.action_by_number_jit(gradient[i], gradient[i], 0, "mul")

        # test_error = 0
        # if len(test_set_index_list) != 0 and hps[2] % p_p == 0:
        #     for test_i in test_set_index_list:
        #         predata_i = np.copy(inputS[test_i])
        #         data_fp = (predata_i, netInfo_, inputs_, outputs_, weights_, False)
        #         forward_propagation(data_fp)
        #         predata_o = np.copy(outputS[test_i])
        #         test_error += Loss_Calculator((predata_o, outputs_[-1], loss_))
        #     error_function_[1][int(hps[2] / p_p)] = test_error / len(test_set_index_list)

        hps[2], hps[0] = hps[2] + 1, hps[0] + 1
        if hps[2] % parts_count == 0:
            np.random.shuffle(train_set_index_list)
            hps[0], hps[1] = 0, hps[1] + 1

def a_loop(input_ : np.ndarray, output_ : np.ndarray):
    gradients = []
    for block in weights:
        gradients.append(np.zeros(block.shape, dtype='float64'))

    data_fp = (input_, netInfo, inputs, outputs, gradients, False)
    # forward_propagation(data_fp)

    print_neuron_info()
    print_weights()

    data_bp = (output_, netInfo, inputs, outputs, weights, gradients, loss)
    back_propagation(data_bp)

    for i, block in enumerate(gradients):
        print("Gradient [{}]:".format(i))
        mx.print_matrix(block, '\t')
    print()


def ADAM(data : tuple):
    i, weights_, momentum1, momentum2, gradient, b1, b2, t, eps, alp = data

    a1, a2, a = np.zeros(weights_[i].shape, dtype='float64'), \
                np.zeros(weights_[i].shape, dtype='float64'), \
                np.zeros(weights_[i].shape, dtype='float64')

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


def Loss_Calculator(data : tuple):
    output_data, last_layer_output, loss_ = data
    if loss_ == Loss.Categorical_cross_entropy:
        return Cross_Entropy((output_data, last_layer_output))
    elif loss_ == Loss.Quadratic_loss:
        return Quadratic_loss((output_data, last_layer_output))


def Cross_Entropy(data : tuple):
    output, last_layer_output = data
    E = 0
    for a_i, a_value in enumerate(last_layer_output):
        if a_value < 0.0000000001:
            E += output[a_i] * (-25)
        else:
            E += output[a_i] * np.log(a_value)
    return -E


def Quadratic_loss(data : tuple):
    output, last_layer_output = data
    E = 0
    for a_i, a_value in enumerate(last_layer_output):
        E += (output[a_i] - a_value) ** 2
    return E
    
# additional functions
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


def jit_equal1d(arr1d_1 : np.ndarray, arr1d_2 : np.ndarray):
    for ind, el in enumerate(arr1d_2):
        arr1d_1[ind] = el


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
