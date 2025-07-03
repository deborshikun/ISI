    # # Perturbed input constraints
    # for t in range(T):
    #     for i in range(input_dim):
    #         solver.add(
    #             And(
    #                 z3_abs(perturbed_x[t][i] - x[t][i]) <= epsilon,
    #                 perturbed_x[t][i] >= 0,
    #                 perturbed_x[t][i] <= 255
    #             )
    #         )
    #         floor_scaled = Real(f"floor_scaled_input_{t}_{i}")
    #         solver.add(floor_scaled == ToReal(ToInt(x[t][i] * scale)))
    #         solver.add(rounded_x[t][i] == If(x[t][i] * scale - floor_scaled >= 0.5, floor_scaled + 1, floor_scaled) / scale)
    #         floor_scaled_perturbed = Real(f"floor_scaled_input_{t}_{i}")
    #         solver.add(floor_scaled_perturbed == ToReal(ToInt(perturbed_x[t][i] * scale)))
    #         solver.add(rounded_x_perturbed[t][i] == If(perturbed_x[t][i] * scale - floor_scaled >= 0.5, floor_scaled + 1, floor_scaled) / scale)

    # Constraint for limiting the number of perturbed frames and pixels
    # solver.add(perturbed_frames == Sum([If(Or(*[perturbed_x[t][i] != x[t][i] for i in range(input_dim)]), 1, 0) for t in range(T)]))
    # solver.add(And(perturbed_frames >= 0, perturbed_frames <= delta))

    # for t in range(T):
    #     solver.add(perturbed_pixels[t] == Sum([If(perturbed_x[t][i] != x[t][i], 1, 0) for i in range(input_dim)]))
    #     solver.add(And(perturbed_pixels[t] >= 0, perturbed_pixels[t] <= pixel_delta))

    # for layer_idx in range(len(hidden_dim) + 1):
    #     input_size = input_dim if layer_idx == 0 else hidden_dim[layer_idx - 1]
    #     output_size = hidden_dim[layer_idx] if layer_idx < len(hidden_dim) else output_dim

    #     prev_activation = rounded_x_perturbed if layer_idx == 0 else rounded_relu_fc[layer_idx - 1]
    #     prev_activation_original = rounded_x if layer_idx == 0 else rounded_relu_fc[layer_idx - 1]
    #     for t in range(T):
    #         for j in range(input_size):
    #             solver.add(delta_fc[layer_idx][t][j] == (prev_activation[t][j] - (prev_activation[t-1][j] if t > 0 else 0)))
    #             solver.add(delta_fc[layer_idx][t][j] <= (prev_activation_original[t][j] - (prev_activation_original[t-1][j] if t > 0 else 0)) + 2 * epsilon)
    #             solver.add(delta_fc[layer_idx][t][j] >= (prev_activation_original[t][j] - (prev_activation_original[t-1][j] if t > 0 else 0)) - 2 * epsilon)
                
    #         for j in range(output_dim):
    #             solver.add(sigma_fc[layer_idx][t][j] == Sum([W_fc[layer_idx][j][i] * delta_fc[layer_idx][t][i] for i in range(input_size)]) + b_fc[layer_idx][j] + (sigma_fc[layer_idx][t-1][j] if t > 0 else 0))
    #             solver.add(sigma_fc[layer_idx][t][j] <= Sum([W_fc[layer_idx][j][i] * (prev_activation_original[t-1][i] + epsilon) for i in range(input_size)]) + b_fc[layer_idx][j] * (t+1))
    #             solver.add(sigma_fc[layer_idx][t][j] >= Sum([W_fc[layer_idx][j][i] * (prev_activation_original[t-1][i] - epsilon) for i in range(input_size)]) + b_fc[layer_idx][j] * (t+1))


    #             if layer_idx < len(hidden_dim):  # Apply ReLU only for hidden layers
    #                 # relu_expr = If(sigma_expr > 0, sigma_expr, 0)
    #                 # print(relu_expr)
    #                 solver.add(relu_fc[layer_idx][t][j] == If(sigma_fc[layer_idx][t][j] > 0, sigma_fc[layer_idx][t][j], 0))

    #                 floor_relu = Real(f"floor_relu_fc{layer_idx}_{t}_{j}")
    #                 solver.add(floor_relu == ToReal(ToInt(relu_fc[layer_idx][t][j] * scale)))
    #                 solver.add(rounded_relu_fc[layer_idx][t][j] == If(relu_fc[layer_idx][t][j] * scale - floor_relu >= 0.5, floor_relu + 1, floor_relu) / scale)

    # Compute predicted labels at each time step
    # for t in range(T):
    #     max_value = Real(f"max_output_{t}")
    #     for j in range(output_dim):
    #         solver.add(max_value >= sigma_fc[-1][t][j])  # max_value is at least as large as every sigma_fc[-1]
    #     solver.add(Or(*[max_value == sigma_fc[-1][t][j] for j in range(output_dim)]))  # max_value equals at least one sigma_fc[-1]

    #     for j in range(output_dim):
    #         solver.add(Implies(sigma_fc[-1][t][j] == max_value, predicted_labels[t] == j))

    # # Majority voting: Final sequence label
    # label_counts = [Int(f"label_count_{j}") for j in range(output_dim)]
    # for j in range(output_dim):
    #     solver.add(label_counts[j] == Sum([If(predicted_labels[t] == j, 1, 0) for t in range(T)]))

    # max_count = Int("max_count")
    # for j in range(output_dim):
    #     solver.add(max_count >= label_counts[j])  # max_count >= all label_counts
    # solver.add(Or(*[max_count == label_counts[j] for j in range(output_dim)]))  # max_count equals at least one label_count
    # solver.add(Or(*[And(label_counts[j] == max_count, final_label == j) for j in range(output_dim)]))

    # # Perturbation goal: Final label does not match original label
    # print(int(label.item()))
    # solver.add(final_label != int(label.item()))
    # # with open("assertions.txt", "w") as f:
    # #     for x in solver.assertions():
    # #         f.write(str(x)+ "\n")
    # print(len(delta_fc))
    # print(len(sigma_fc))

    # # print(solver.assertions())
    # # print(solver.get_unsat_core())
    # # Check satisfiability
    # result = solver.check()
    # if result == sat:
    #     model = solver.model()
    #     print("Model is SATISFIABLE!")
    #     print(model[final_label])
    #     delta_val = model[delta].as_long()
    #     pixel_delta_val = model[pixel_delta].as_long()
    #     epsilon_val = model[epsilon].as_long()
    #     # with open("model.txt", "w") as f:
    #     #     for x in model:
    #     #         f.write(str(x)+ "\n")
    #     print(f"Adversarial example found with (\u0394, \u03B4, \u03B5): ({delta_val}, {pixel_delta_val}, {epsilon_val})")
    #     end_time = time.time()  # Stop timing
    #     time_taken = (end_time - start_time) / 60  # Convert seconds to minutes

    #     with open("all_4.txt", "a") as f:
    #         f.write(f"Adversarial example found with (��, ��, ��): ({delta_val}, {pixel_delta_val}, {epsilon_val})\n")
    #         f.write(f"Time taken: {time_taken:.6f} minutes\n")

    # elif result == unsat:
    #     print("No adversarial example found for this sequence.")
    #     end_time = time.time()  # Stop timing
    #     time_taken = (end_time - start_time) / 60  # Convert seconds to minutes

    #     with open("all_4.txt", "a") as f:
    #         f.write("No adversarial example found for this sequence.\n")
    #         f.write(f"Time taken: {time_taken:.6f} minutes\n")
    
    # else:
    #     reason = solver.reason_unknown()
    #     if "timeout" in reason:
    #         print("Solver has reached timeout.")
    #         with open("all_4.txt", "a") as f:
    #             f.write("Solver has reached timeout.\n")
    #     else:
    #         print(f"Solver returned unknown due to: {reason}")
    #         with open("all_4.txt", "a") as f:
    #             f.write(f"Solver returned unknown due to: {reason}\n")






from z3 import *
import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import time
import numpy as np

# Helper function for Z3 absolute value
def z3_abs(expr):
    return If(expr >= 0, expr, -expr)

# Load weights from ONNX model
def load_onnx_weights(onnx_model_path):
    model = onnx.load(onnx_model_path)
    weights = {}
    for initializer in model.graph.initializer:
        weights[initializer.name] = numpy_helper.to_array(initializer)
    return weights

def get_model_dimensions(onnx_model_path):
    # Load the ONNX model
    model = onnx.load(onnx_model_path)
    
    # Extract input dimensions
    input_dims = [
        [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        for inp in model.graph.input
    ]
    
    # Extract output dimensions
    output_dims = [
        [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        for out in model.graph.output
    ]
    
    # Extract hidden layer dimensions (weights/biases of layers)
    hidden_layer_dims = []
    for initializer in model.graph.initializer:
        hidden_layer_dims.append(list(initializer.dims))

    input_dims = input_dims[0][-1] * input_dims[0][-2]
    i = 1
    n = len(hidden_layer_dims)//2
    hidden_dims = []
    print(hidden_layer_dims)
    while i <= n-1:
        hidden_dims.append(hidden_layer_dims[2 * i -1][0])      # hidden_layer_dims contains [[weight_dims], [bias_dim] for each of the hidden layer]
        i = i + 1
    output_dims = output_dims[0][-1]
    
    return input_dims, hidden_dims, output_dims

# Temporal MNIST Dataset
class TemporalMNISTDataset(Dataset):
    def __init__(self, sequences, labels, transform=None):
        self.sequences = [np.array(seq) for seq in sequences]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # print(sequence[0])
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure label is LongTensor

        if self.transform:
            sequence = torch.stack([self.transform(img) for img in sequence], dim=0)

        return sequence, label

# Load Temporal MNIST dataset
with open("test_sequences_8.pkl", "rb") as file:
    data = pickle.load(file)

test_data = TemporalMNISTDataset(data["test"]["sequences"], data["test"]["labels"])
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Load ONNX model weights
onnx_path = "temporal_two_layer_dnn.onnx"
weights = load_onnx_weights(onnx_path)

# Initialize Z3 solver
solver = Solver()

# Dimensions
T = int(input("Enter the number of time steps(sequence length): "))  # Number of time steps (sequence length)
Delta = int(input("Enter the percentage of Partial Frame Drop: "))
Pixel_Delta = int(input("Enter the percentage of Partial Frame Perturbation: "))
Epsilon = int(input("Enter the percentage of Partial Pixel Perturbation: "))
# input_dim = 28 * 28  # Input dimension (flattened MNIST image)
# hidden_dim = 128     # Hidden layer size
# output_dim = 10      # Output layer size

# Get the dimensions
input_dim, hidden_dim, output_dim = get_model_dimensions(onnx_path)

print("Input Dimensions:", input_dim)
print("Hidden Layer Dimensions:", hidden_dim)
print("Output Dimensions:", output_dim)



# Define parameters as Z3 variables
scale = Real("scale")          # Scaling factor for rounding
delta = Int("delta")            # Number of frames to perturb
pixel_delta = Int("pixel_delta")  # Number of pixels to perturb in each frame
epsilon = Int("epsilon")        # Magnitude of perturbation

perturbed_frames = Int("perturbed_frames")
perturbed_pixels = [Int("perturbed_pixels_{t}") for t in range(T)]

# Variables for original inputs, perturbed inputs, deltas, sigmas, activations, and rounded values
x = [[Int(f"x_{t}_{i}") for i in range(input_dim)] for t in range(T)]  # Inputs
perturbed_x = [[Int(f"perturbed_x_{t}_{i}") for i in range(input_dim)] for t in range(T)]
rounded_x = [[Real(f"rounded_x_{t}_{i}") for i in range(input_dim)] for t in range(T)]
rounded_x_perturbed = [[Real(f"rounded_x_perturbed_{t}_{i}") for i in range(input_dim)] for t in range(T)]


delta_fc = []
sigma_fc = []
relu_fc = []
rounded_relu_fc = []

for layer_idx in range(len(hidden_dim) + 1):
    input_size = input_dim if layer_idx == 0 else hidden_dim[layer_idx - 1]
    output_size = hidden_dim[layer_idx] if layer_idx < len(hidden_dim) else output_dim

    delta_layer = [[Real(f"delta_fc{layer_idx}_{t}_{j}") for j in range(input_size)] for t in range(T)]
    sigma_layer = [[Real(f"sigma_fc{layer_idx}_{t}_{j}") for j in range(output_size)] for t in range(T)]
    relu_layer = [[Real(f"relu_fc{layer_idx}_{t}_{j}") for j in range(output_size)] for t in range(T)]
    rounded_layer = [[Real(f"rounded_relu_fc{layer_idx}_{t}_{j}") for j in range(output_size)] for t in range(T)]

    delta_fc.append(delta_layer)
    sigma_fc.append(sigma_layer)
    relu_fc.append(relu_layer)
    rounded_relu_fc.append(rounded_layer)

# Variables for predicted labels
predicted_labels = [Int(f"predicted_label_{t}") for t in range(T)]

# Majority voting label
final_label = Int("final_label")

model_time = []

# Processing each sequence
for idx, (sequences, labels) in enumerate(test_loader):
    start_time = time.time()
    
    solver.reset()
    solver.set("timeout", 3600000)  # Timeout to prevent excessive solver time
    # Bounds for delta, pixel_delta, and epsilon
    solver.add(And(delta >= 1, delta <= T))  # delta can vary from 1 to the maximum sequence length
    solver.add(And(pixel_delta >= 0, pixel_delta <= input_dim))  # pixel_delta ranges up to total pixels in an image
    solver.add(And(epsilon >= 0, epsilon <= 255))  # epsilon must be positive and within valid pixel intensity range

    solver.add(delta == ToInt(T * (RealVal(Delta) / RealVal(100)) + RealVal(0.5)))
    solver.add(pixel_delta == ToInt(input_dim * (RealVal(Pixel_Delta) / RealVal(100)) + RealVal(0.5)))
    solver.add(epsilon == ToInt(255 * (RealVal(Epsilon) / RealVal(100)) + RealVal(0.5)))
    solver.add(scale == 1.0)
    
    x_data = [[x.view(-1, input_dim)] for x in sequences]
    label = labels

    # Dynamically create Z3 variables for weights and biases
    W_fc = []
    b_fc = []

    for layer_idx in range(len(hidden_dim) + 1):
        input_size = input_dim if layer_idx == 0 else hidden_dim[layer_idx - 1]
        output_size = hidden_dim[layer_idx] if layer_idx < len(hidden_dim) else output_dim

        W_layer = [[Real(f"W_fc{layer_idx}_{j}_{i}") for i in range(input_size)] for j in range(output_size)]
        b_layer = [Real(f"b_fc{layer_idx}_{j}") for j in range(output_size)]

        # Load ONNX weights and add constraints to the solver
        weight_name = f"fc{layer_idx + 1}.weight"
        bias_name = f"fc{layer_idx + 1}.bias"
        # print("Size of weight",torch.from_numpy(weights[weight_name]).float().size())
        # print(torch.from_numpy(weights[weight_name]).float())
        for j in range(output_size):
            for i in range(input_size):
                solver.add(W_layer[j][i] == float(weights[weight_name][j][i]))
            solver.add(b_layer[j] == float(weights[bias_name][j]))

        W_fc.append(W_layer)
        b_fc.append(b_layer)

    # Input constraints
    for t in range(T):
        for i in range(input_dim):
            solver.add(x[t][i] == x_data[0][0][t][i].item())

    print(f"\nProcessing Sequence {idx + 1}...")

    # Avoid unnecessary constraints when Delta is 0
    if round(T * (Delta / 100)) > 0:
        for t in range(T):
            for i in range(input_dim):
                solver.add(
                    And(
                        z3_abs(perturbed_x[t][i] - x[t][i]) <= epsilon,
                        perturbed_x[t][i] >= 0,
                        perturbed_x[t][i] <= 255
                    )
                )

                solver.add(rounded_x[t][i] == ToInt(x[t][i] * scale + 0.5))
                solver.add(rounded_x_perturbed[t][i] == ToInt(perturbed_x[t][i] * scale + 0.5))
    else:
        for t in range(T):
            for i in range(input_dim):
                solver.add(rounded_x[t][i] == ToInt(x[t][i] * scale + 0.5))
                solver.add(rounded_x_perturbed[t][i] == ToInt(perturbed_x[t][i] * scale + 0.5))

    # Bounding perturbed frames to prevent overflow
    if round(T * (Delta / 100)) > 0:
        # Constraint for limiting the number of perturbed frames and pixels
        solver.add(perturbed_frames == Sum([If(Or(*[perturbed_x[t][i] != x[t][i] for i in range(input_dim)]), 1, 0) for t in range(T)]))
        solver.add(And(perturbed_frames >= 0, perturbed_frames <= delta))

        for t in range(T):
            solver.add(perturbed_pixels[t] == Sum([If(perturbed_x[t][i] != x[t][i], 1, 0) for i in range(input_dim)]))
            solver.add(And(perturbed_pixels[t] >= 0, perturbed_pixels[t] <= pixel_delta))
    
    # Compute max absolute weight and bias
    # max_abs_weight = max(abs(float(w)) for weight_matrix in weights.values() if isinstance(weight_matrix, np.ndarray) for w in weight_matrix.flatten())
    # max_abs_bias = max(abs(float(b)) for bias_vector in weights.values() if isinstance(bias_vector, np.ndarray) for b in bias_vector.flatten())

    # Compute weight and bias bounds dynamically
    # max_abs_weight = max(abs(float(w)) for weight_matrix in weights.values() for row in weight_matrix for w in row)
    # max_abs_bias = max(abs(float(b)) for bias_vector in weights.values() for b in bias_vector)

    # sigma_upper_bound = input_dim * max_abs_weight * 255 + max_abs_bias
    # sigma_lower_bound = -sigma_upper_bound  # Symmetric bounds  
    # print(sigma_upper_bound)
    
    # Hidden layers constraints
    for layer_idx in range(len(hidden_dim) + 1):
        input_size = input_dim if layer_idx == 0 else hidden_dim[layer_idx - 1]
        output_size = hidden_dim[layer_idx] if layer_idx < len(hidden_dim) else output_dim

        # prev_activation = rounded_x if layer_idx == 0 else rounded_relu_fc[layer_idx - 1]
        prev_activation = rounded_x if layer_idx == 0 else rounded_relu_fc[layer_idx - 1]
        prev_activation_original = rounded_x if layer_idx == 0 else rounded_relu_fc[layer_idx - 1]

        for t in range(T):
            for j in range(input_size):
                solver.add(delta_fc[layer_idx][t][j] == (prev_activation[t][j] - (prev_activation[t-1][j] if t > 0 else 0)))
                
            for j in range(output_size):
                solver.add(
                    sigma_fc[layer_idx][t][j] == 
                    Sum([W_fc[layer_idx][j][i] * delta_fc[layer_idx][t][i] for i in range(input_size)]) +
                    b_fc[layer_idx][j] + (sigma_fc[layer_idx][t - 1][j] if t > 0 else 0)
                )
                solver.add(sigma_fc[layer_idx][t][j] <= Sum([W_fc[layer_idx][j][i] * (prev_activation_original[t-1][i] + epsilon) for i in range(input_size)]) + b_fc[layer_idx][j] * (t+1))
                solver.add(sigma_fc[layer_idx][t][j] >= Sum([W_fc[layer_idx][j][i] * (prev_activation_original[t-1][i] - epsilon) for i in range(input_size)]) + b_fc[layer_idx][j] * (t+1))
                # solver.add(sigma_fc[layer_idx][t][j] <= sigma_upper_bound)
                # solver.add(sigma_fc[layer_idx][t][j] >= sigma_lower_bound)


                # Prevent sigma overflow by bounding
                # solver.add(sigma_fc[layer_idx][t][j] <= 1e6)
                # solver.add(sigma_fc[layer_idx][t][j] >= -1e6)

                if layer_idx < len(hidden_dim):  # Apply ReLU only for hidden layers
                    solver.add(relu_fc[layer_idx][t][j] == If(sigma_fc[layer_idx][t][j] > 0, sigma_fc[layer_idx][t][j], 0))
                    solver.add(rounded_relu_fc[layer_idx][t][j] == ToInt(relu_fc[layer_idx][t][j] * scale + 0.5))

    # Compute predicted labels at each time step
    for t in range(T):
        max_value = Real(f"max_output_{t}")
        
        # Ensure max_value is at least as large as any output neuron
        for j in range(output_dim):
            solver.add(max_value >= sigma_fc[-1][t][j])  
        
        # Ensure max_value corresponds to at least one output neuron
        solver.add(Or(*[max_value == sigma_fc[-1][t][j] for j in range(output_dim)]))  

        # Assign predicted label based on max_value
        for j in range(output_dim):
            solver.add(Implies(sigma_fc[-1][t][j] == max_value, predicted_labels[t] == j))

    
    label_counts = [Int(f"label_counts_{j}") for j in range(output_dim)]
    
    # Avoid excessively large majority voting computations
    for j in range(output_dim):
        solver.add(label_counts[j] == Sum([If(predicted_labels[t] == j, 1, 0) for t in range(T)]))
        solver.add(label_counts[j] <= T)

    # Final label selection
    max_count = Int("max_count")
    for j in range(output_dim):
        solver.add(max_count >= label_counts[j])  # max_count >= all label_counts
    solver.add(Or(*[max_count == label_counts[j] for j in range(output_dim)]))  # max_count equals at least one label_count
    solver.add(Or(*[And(label_counts[j] == max_count, final_label == j) for j in range(output_dim)]))

    # Goal: Find an adversarial example
    print("Original label: ",int(label.item()))
    solver.add(final_label != int(label.item()))

    result = solver.check()
    if result == sat:
        print("Model is SATISFIABLE!")
        end_time = time.time()  # Stop timing
        time_taken = (end_time - start_time)
        model_time.append(time_taken)
        model = solver.model()
        print("Countrer example label: ",model[final_label])
        delta_val = model[delta].as_long()
        pixel_delta_val = model[pixel_delta].as_long()
        epsilon_val = model[epsilon].as_long()

        with open(f"model_exp3_{T}_{Delta}_{Pixel_Delta}_{Epsilon}.txt", "a") as f:
            f.write(f"Adversarial example found with (��, ��, ��): ({delta_val}, {pixel_delta_val}, {epsilon_val})\n")
            f.write(f"Time taken: {time_taken:.6f} seconds\n")

    elif result == unsat:
        print("No adversarial example found for this sequence.")
        end_time = time.time()  # Stop timing
        time_taken = (end_time - start_time)
        model_time.append(time_taken)

        with open(f"model_exp3_{T}_{Delta}_{Pixel_Delta}_{Epsilon}.txt", "a") as f:
            f.write("No adversarial example found for this sequence.\n")
            f.write(f"Time taken: {time_taken:.6f} seconds\n")
    
    else:
        reason = solver.reason_unknown()
        if "timeout" in reason:
            print("Solver has reached timeout.")
            with open(f"model_exp3_{T}_{Delta}_{Pixel_Delta}_{Epsilon}.txt", "a") as f:
                f.write("Solver has reached timeout.\n")
        else:
            print(f"Solver returned unknown due to: {reason}")
            with open(f"model_exp3_{T}_{Delta}_{Pixel_Delta}_{Epsilon}.txt", "a") as f:
                f.write(f"Solver returned unknown due to: {reason}\n")

with open(f"model_exp3_{T}_{Delta}_{Pixel_Delta}_{Epsilon}.txt", "a") as f:
    f.write(f"Average time: {sum(model_time)/len(model_time)}\n")
    