import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5 #eta

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

#Wj
weights_hidden_output = np.array([0.1, -0.3])

## Forward pass
hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)

output_layer_in = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_in)

## Backwards pass
## TODO: Calculate error
#  (y - y')
error = target-output

# TODO: Calculate error gradient for output layer
# delta_salida = (y - y')f'(h)
del_err_output = error * sigmoid_prime(output_layer_in)

# TODO: Calculate error gradient for hidden layer
#propagate the error to hidden layer
# delta_jh = delta_salida * Wj *  f'(hj)
del_err_hidden = (del_err_output * (weights_hidden_output * sigmoid_prime(hidden_layer_input)))

# TODO: Calculate change in weights for hidden layer to output layer
#deltaW = eta * delta output * xj                                       Gradient descent step
delta_w_h_o = learnrate * del_err_output * hidden_layer_output

# TODO: Calculate change in weights for input layer to hidden layer
#deltaW = eta * delta hidden * xi                                       Gradient descent step
delta_w_i_h = learnrate * (del_err_hidden * x[:,None])

print('Change in weights for hidden layer to output layer:')
print(delta_w_h_o)
print('Change in weights for input layer to hidden layer:')
print(delta_w_i_h)
