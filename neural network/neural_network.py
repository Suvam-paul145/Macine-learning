import math
import random

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    sx = sigmoid(x)
    return sx * (1 - sx)

# Initialize weights and biases randomly
def init_network():
    network = {
        'input_hidden_weights': [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)],  # 2x2 weights
        'hidden_biases': [random.uniform(-1, 1) for _ in range(2)],
        'hidden_output_weights': [random.uniform(-1, 1) for _ in range(2)],
        'output_bias': random.uniform(-1, 1)
    }
    return network

# Forward pass
def forward_pass(network, inputs):
    z_hidden = []
    a_hidden = []

    # Hidden layer
    for i in range(2):
        z = sum(inputs[j] * network['input_hidden_weights'][i][j] for j in range(2)) + network['hidden_biases'][i]
        z_hidden.append(z)
        a_hidden.append(sigmoid(z))

    # Output layer
    z_output = sum(a_hidden[i] * network['hidden_output_weights'][i] for i in range(2)) + network['output_bias']
    output = sigmoid(z_output)

    return {
        'inputs': inputs,
        'z_hidden': z_hidden,
        'a_hidden': a_hidden,
        'z_output': z_output,
        'output': output
    }

# Backward pass (manual gradient descent)
def backward_pass(network, forward, target, learning_rate=0.1):
    # Output error
    error = forward['output'] - target
    d_output = error * sigmoid_derivative(forward['z_output'])

    # Gradients for hidden-output weights and bias
    for i in range(2):
        network['hidden_output_weights'][i] -= learning_rate * d_output * forward['a_hidden'][i]
    network['output_bias'] -= learning_rate * d_output

    # Gradients for input-hidden weights and biases
    for i in range(2):
        d_hidden = d_output * network['hidden_output_weights'][i] * sigmoid_derivative(forward['z_hidden'][i])
        for j in range(2):
            network['input_hidden_weights'][i][j] -= learning_rate * d_hidden * forward['inputs'][j]
        network['hidden_biases'][i] -= learning_rate * d_hidden


# Train the network
def train(network, data, epochs=1000):
    for epoch in range(epochs):
        total_loss = 0
        for inputs, target in data:
            fwd = forward_pass(network, inputs)
            backward_pass(network, fwd, target)
            total_loss += (fwd['output'] - target) ** 2
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(data):.4f}")

# Predict
def predict(network, inputs):
    fwd = forward_pass(network, inputs)
    return fwd['output']

# Example training data for XOR (non-linearly separable)
training_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# Initialize and train
nn = init_network()
train(nn, training_data)

# Test
print("Predictions:")
for inputs, _ in training_data:
    output = predict(nn, inputs)
    print(f"Input: {inputs} -> Output: {round(output, 3)}")
