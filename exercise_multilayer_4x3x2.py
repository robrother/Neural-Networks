import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

# Network size
N_input = 4   # input layer
N_hidden = 3  # hidden layer 
N_output = 2  # output layer

np.random.seed(42)
# Make some fake data
X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# TODO: Make a forward pass through the network

#------------------HIDDEN LAYER------------------------------------------------------------------------------
#Calculamos hj = wij * xi + b
hidden_layer_in = np.dot(X, weights_input_to_hidden)
#calculamos la salida aj = sigmoid (hj) 
hidden_layer_out = [sigmoid(hidden_layer_in[0]), sigmoid(hidden_layer_in[1]), sigmoid(hidden_layer_in[2])]
#Imprimimos los resultados
print('Hidden-layer Output:')
print(hidden_layer_out)

#------------------OUTPUT LAYER------------------------------------------------------------------------------
#Calculamos hK = wjk * aj + b
output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
#output = sigmoid(Sum (wj k * aj + b)) = sigmoid (hk)
output_layer_out = [sigmoid(output_layer_in[0]), sigmoid(output_layer_in[1])]
#Imprimimos los resultados
print('Output-layer Output:')
print(output_layer_out)
