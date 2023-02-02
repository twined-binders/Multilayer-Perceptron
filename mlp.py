import math

# Input
x1 = 0.5
x2 = 0.8

# Bias
b1 = 0.1
b2 = 0.2
b3 = 0.3

# Bobot
w11 = 0.4
w12 = 0.5
w21 = 0.6
w22 = 0.7
w31 = 0.8
w32 = 0.9

# Target output
target_output = 1

# Learning rate
learning_rate = 0.1

# Jumlah Training Looping (Epoch)
training = 20000

# Aktivasi Sigmoid 
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

for i in range(training):
    # Hidden layer
    h1 = sigmoid(x1*w11 + x2*w12 + b1)
    h2 = sigmoid(x1*w21 + x2*w22 + b2)

    # Output
    output = sigmoid(h1*w31 + h2*w32 + b3)

    # Error
    error = target_output - output

    # Backpropagation
    w31 += learning_rate * error * h1
    w32 += learning_rate * error * h2
    b3 += learning_rate * error
    w11 += learning_rate * error * x1
    w12 += learning_rate * error * x1
    w21 += learning_rate * error * x2
    w22 += learning_rate * error * x2
    b1 += learning_rate * error
    b2 += learning_rate * error

print("Input: x1 = {}, x2 = {}".format(x1, x2))
print("Bias: b1 = {}, b2 = {}, b3 = {}".format(b1, b2, b3))
print("Bobot: w11 = {}, w12 = {}, w21 = {}, w22 = {}, w31 = {}, w32 = {}".format(w11, w12, w21, w22, w31, w32))
print("Hidden layer: h1 = {}, h2 = {}".format(h1, h2))
print("Output: {}".format(output))
