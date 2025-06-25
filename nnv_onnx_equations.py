import onnx
from onnx import numpy_helper, shape_inference
import numpy as np
import os # For checking file existence with input()

def get_tensor_shape(graph, tensor_name_onnx, value_infos_map):
    """Helper to get the shape of an ONNX tensor."""
    if tensor_name_onnx in value_infos_map:
        info = value_infos_map[tensor_name_onnx]
        if info.type.tensor_type.HasField("shape"):
            shape = []
            for dim in info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else: # dim_param or 0 (unknown)
                    shape.append(None)
            return shape
    return None


def generate_equations(onnx_model_path: str):

    if not os.path.exists(onnx_model_path):
        print(f"Error: The file '{onnx_model_path}' was not found.")
        return
    try:
        model = onnx.load(onnx_model_path)
    except Exception as e:
        print(f"Error loading ONNX model '{onnx_model_path}': {e}")
        return

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Error during shape inference for '{onnx_model_path}': {e}. Proceeding with potentially incomplete shape info.")

    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    # Build a quick lookup for value_infos (includes inputs, outputs, and intermediate tensors with shapes)
    value_infos_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}

    equations = []
    # Maps ONNX tensor name to its final readable symbolic name (e.g., "x1", "h1_w", "h2_a")
    symbolic_name_map: dict[str, str] = {}

    x_idx = 1 # Counter for graph inputs x1, x2...
    h_idx = 1 # Counter for hidden/intermediate tensors h1, h2...

    # 1. Assign symbolic names to graph inputs
    for inp_info in graph.input:
        if inp_info.name not in initializers: # Actual inputs
            symbolic_name_map[inp_info.name] = f"x{x_idx}"
            x_idx += 1

    # Pre-scan for activation functions that immediately follow a Gemm
    # Maps the ONNX output name of a Gemm to the (op_type, onnx_output_name_of_activation_node)
    direct_activations_info: dict[str, tuple[str, str]] = {}
    node_map = {node.name if node.name else node.output[0] : node for node in graph.node} # For lookup if needed

    # Create a map from tensor name to the node that produces it
    producer_map = {}
    for node in graph.node:
        for out_name in node.output:
            producer_map[out_name] = node

    for node in graph.node:
        if node.op_type in ('Relu', 'Sigmoid', 'Tanh') and node.input:
            input_to_act_onnx = node.input[0]
            if input_to_act_onnx in producer_map:
                producer_node = producer_map[input_to_act_onnx]
                if producer_node.op_type == 'Gemm':
                    # Gemm output -> (ActivationOpType, ActivationOutputONNXName)
                    direct_activations_info[input_to_act_onnx] = (node.op_type, node.output[0])

    # 2. Iterate through nodes to generate equations and assign symbolic names
    for node_idx, node in enumerate(graph.node):
        op_type = node.op_type
        inputs_onnx = node.input
        outputs_onnx = node.output

        if not outputs_onnx: continue # Skip nodes with no output

        output_onnx_name = outputs_onnx[0] # Primary output for naming

        # Fetch readable symbols for inputs for the equation's RHS
        eq_rhs_input_symbols = []
        for inp_name_onnx in inputs_onnx:
            if inp_name_onnx in symbolic_name_map:
                eq_rhs_input_symbols.append(symbolic_name_map[inp_name_onnx])
            elif inp_name_onnx in initializers:
                # Could assign W_i, B_i names to initializers too, for now use raw
                eq_rhs_input_symbols.append(inp_name_onnx)
            else:
                # This might happen for optional inputs or if an input is not yet processed (should not for valid graph)
                eq_rhs_input_symbols.append(f"UNRESOLVED({inp_name_onnx})")

        # Determine and assign symbolic name for the current node's output (LHS)
        current_lhs_symbol = ""
        created_new_h_base_for_gemm = False # Flag to manage h_idx increment for Gemm+Act pairs

        if op_type == 'Gemm':
            base_h_name = f"h{h_idx}"
            created_new_h_base_for_gemm = True # This Gemm will use h_idx

            current_lhs_symbol = f"{base_h_name}_w"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol

            # --- Gemm Equation Generation ---
            X_sym = eq_rhs_input_symbols[0]
            W_name = inputs_onnx[1] # ONNX name of weight tensor
            B_name = inputs_onnx[2] if len(inputs_onnx) > 2 else None

            if W_name not in initializers:
                equations.append(f"{current_lhs_symbol} = Error_Weight_Not_Found({W_name})")
                if created_new_h_base_for_gemm: h_idx +=1 # Consume h_idx even on error
                continue

            W = initializers[W_name]
            B = initializers.get(B_name) if B_name else None

            transB = next((attr.i for attr in node.attribute if attr.name == 'transB'), 1) # Default is 1
            alpha = next((attr.f for attr in node.attribute if attr.name == 'alpha'), 1.0)
            beta = next((attr.f for attr in node.attribute if attr.name == 'beta'), 1.0)

            _W_matrix = W if transB else W.T # Effective weight matrix for W[j,i] indexing

            out_features = _W_matrix.shape[0]
            in_features_from_weight = _W_matrix.shape[1]

            # Get shape of input X to determine if it's a vector for indexing
            # Use original ONNX name of X for shape lookup
            X_onnx_name = inputs_onnx[0]
            X_shape = get_tensor_shape(graph, X_onnx_name, value_infos_map)

            # Determine if X_sym needs indexing (e.g., if X_sym is "x1" or "h1_a")
            # X_sym is already the readable name.
            # A simple heuristic: if X_shape suggests multiple features.
            # This assumes features are in the last non-batch dimension.
            num_input_elements_for_X = 1
            is_X_vector_like = False
            if X_shape:
                # If X_shape is [batch_dim, features] or [features]
                if len(X_shape) > 1 and X_shape[1] is not None and X_shape[1] > 1 : # Batched features
                    num_input_elements_for_X = X_shape[1]
                    is_X_vector_like = True
                elif len(X_shape) == 1 and X_shape[0] is not None and X_shape[0] > 1: # Unbatched features
                    num_input_elements_for_X = X_shape[0]
                    is_X_vector_like = True

            # Ensure in_features_from_weight matches num_input_elements_for_X if X is vector-like
            # This is a sanity check; ONNX model should be valid.

            for j in range(out_features):
                terms = []
                for i in range(in_features_from_weight): # Iterate over input features
                    weight_val = _W_matrix[j,i]
                    if abs(weight_val * alpha) < 1e-9: continue

                    input_element_sym = X_sym
                    if is_X_vector_like : # If X is a vector like x1 or h1_a, index it
                        input_element_sym = f"{X_sym}[{i}]"

                    term_str = f"{alpha*weight_val:.4f}*{input_element_sym}" if alpha !=1 else f"{weight_val:.4f}*{input_element_sym}"
                    terms.append(term_str)

                eq_rhs = " + ".join(terms) if terms else "0"
                if B is not None and abs(beta) > 1e-9:
                    bias_val = B[j]
                    if abs(beta * bias_val) > 1e-9 or not terms:
                        eq_rhs += f" + {beta*bias_val:.4f}"

                lhs_indexed = f"{current_lhs_symbol}[{j}]" if out_features > 1 else current_lhs_symbol
                equations.append(f"{lhs_indexed} = {eq_rhs}")
            # --- End Gemm Equation Generation ---

            if output_onnx_name in direct_activations_info:
                act_op_type, act_output_onnx_name = direct_activations_info[output_onnx_name]

                post_act_lhs_symbol = f"{base_h_name}_a" # Use same base_h_name
                symbolic_name_map[act_output_onnx_name] = post_act_lhs_symbol

                # --- Activation Equation Generation (for Gemm pair) ---
                # Input to activation is current_lhs_symbol (e.g. hK_w)
                # Output is post_act_lhs_symbol (e.g. hK_a)
                # Need to consider if current_lhs_symbol (gemm output) is scalar or vector
                gemm_out_shape = get_tensor_shape(graph, output_onnx_name, value_infos_map)
                num_elements_act = 1
                is_gemm_out_vector = False
                if gemm_out_shape:
                    last_dim = gemm_out_shape[-1]
                    if last_dim is not None and last_dim > 1:
                        num_elements_act = last_dim
                        is_gemm_out_vector = True
                    elif len(gemm_out_shape) == 1 and gemm_out_shape[0] is not None and gemm_out_shape[0] > 1:
                        num_elements_act = gemm_out_shape[0]
                        is_gemm_out_vector = True

                for j_act in range(num_elements_act):
                    lhs_act_indexed = f"{post_act_lhs_symbol}[{j_act}]" if is_gemm_out_vector else post_act_lhs_symbol
                    rhs_act_indexed = f"{current_lhs_symbol}[{j_act}]" if is_gemm_out_vector else current_lhs_symbol
                    equations.append(f"{lhs_act_indexed} = {act_op_type}({rhs_act_indexed})")
                # --- End Activation Equation Generation ---

            if created_new_h_base_for_gemm: # h_idx was used for this Gemm (and its optional direct activation)
                h_idx += 1


        elif op_type in ('Relu', 'Sigmoid', 'Tanh'):
            # Check if this activation node's output symbol was already defined (e.g., by Gemm's direct activation handling)
            if output_onnx_name in symbolic_name_map:
                # This means it was likely handled as part of a Gemm->Activation pair.
                # The LHS symbol current_lhs_symbol is already set by that logic.
                # No new equation or h_idx increment needed here.
                continue
            else:
                # This is a "standalone" activation (not immediately processed with a Gemm)
                input_symbol_for_act_rhs = eq_rhs_input_symbols[0]

                base_name_for_output_a = ""

                if input_symbol_for_act_rhs.endswith("_w"): # e.g. input hK_w
                    base_name_for_output_a = input_symbol_for_act_rhs[:-2] # -> base hK
                elif input_symbol_for_act_rhs.startswith("h") and not input_symbol_for_act_rhs.endswith("_a"): # e.g. input hK
                    base_name_for_output_a = input_symbol_for_act_rhs # -> base hK
                else: # Input is xK, or hK_a, or something else. Needs a new h_idx.
                    base_name_for_output_a = f"h{h_idx}"
                    h_idx += 1

                current_lhs_symbol = f"{base_name_for_output_a}_a"
                symbolic_name_map[output_onnx_name] = current_lhs_symbol

                # --- Standalone Activation Equation Generation ---
                act_input_onnx_name = inputs_onnx[0]
                act_input_shape = get_tensor_shape(graph, act_input_onnx_name, value_infos_map)
                num_elements_act = 1
                is_act_input_vector = False
                if act_input_shape:
                    last_dim = act_input_shape[-1]
                    if last_dim is not None and last_dim > 1:
                        num_elements_act = last_dim; is_act_input_vector = True
                    elif len(act_input_shape) == 1 and act_input_shape[0] is not None and act_input_shape[0] > 1:
                        num_elements_act = act_input_shape[0]; is_act_input_vector = True

                for j_act in range(num_elements_act):
                    lhs_act_indexed = f"{current_lhs_symbol}[{j_act}]" if is_act_input_vector else current_lhs_symbol
                    rhs_act_indexed = f"{input_symbol_for_act_rhs}[{j_act}]" if is_act_input_vector and "[" not in input_symbol_for_act_rhs else input_symbol_for_act_rhs
                    equations.append(f"{lhs_act_indexed} = {op_type}({rhs_act_indexed})")
                # --- End Standalone Activation Equation ---

        elif op_type in ('Add', 'Mul', 'Sub', 'Div'):
            current_lhs_symbol = f"h{h_idx}"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            h_idx += 1

            # --- Element-wise Binary Op Equation ---
            op_out_shape = get_tensor_shape(graph, output_onnx_name, value_infos_map)
            num_elements_op = 1
            is_op_out_vector = False
            if op_out_shape:
                last_dim = op_out_shape[-1]
                if last_dim is not None and last_dim > 1: num_elements_op = last_dim; is_op_out_vector = True
                elif len(op_out_shape) == 1 and op_out_shape[0] is not None and op_out_shape[0] > 1: num_elements_op = op_out_shape[0]; is_op_out_vector = True

            op_char = {'Add': '+', 'Mul': '*', 'Sub': '-', 'Div': '/'}.get(op_type, f" {op_type} ")

            for j_op in range(num_elements_op):
                lhs_op_indexed = f"{current_lhs_symbol}[{j_op}]" if is_op_out_vector else current_lhs_symbol

                # Handle broadcasting for A and B inputs if output is vector
                rhs_A_sym = eq_rhs_input_symbols[0]
                A_onnx_name = inputs_onnx[0]
                A_shape = get_tensor_shape(graph, A_onnx_name, value_infos_map)
                is_A_scalar_like = not A_shape or (A_shape[-1] == 1 if A_shape else False) or (len(A_shape) == 1 and A_shape[0] == 1)
                rhs_A_indexed = f"{rhs_A_sym}[{j_op}]" if is_op_out_vector and not is_A_scalar_like and "[" not in rhs_A_sym else rhs_A_sym

                rhs_B_sym = eq_rhs_input_symbols[1]
                B_onnx_name = inputs_onnx[1]
                B_shape = get_tensor_shape(graph, B_onnx_name, value_infos_map)
                is_B_scalar_like = not B_shape or (B_shape[-1] == 1 if B_shape else False) or (len(B_shape) == 1 and B_shape[0] == 1)
                rhs_B_indexed = f"{rhs_B_sym}[{j_op}]" if is_op_out_vector and not is_B_scalar_like and "[" not in rhs_B_sym else rhs_B_sym

                equations.append(f"{lhs_op_indexed} = {rhs_A_indexed} {op_char} {rhs_B_indexed}")
            # --- End Element-wise Op ---

        elif op_type in ('Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose', 'Identity'): # Shape or identity ops
            input_sym_for_shape_op = eq_rhs_input_symbols[0]
            # Assign a new h_k name to the output of this shape op for traceability
            current_lhs_symbol = f"h{h_idx}"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            h_idx += 1

            other_inputs_str = ""
            if len(eq_rhs_input_symbols) > 1: # Some ops like Reshape take a shape tensor as input
                other_inputs_str = ", " + ", ".join(eq_rhs_input_symbols[1:])

            equations.append(f"{current_lhs_symbol} = {op_type}({input_sym_for_shape_op}{other_inputs_str})  # Value is based on {input_sym_for_shape_op}")

        elif op_type == 'Constant':
            val_attr = next((attr for attr in node.attribute if attr.name == 'value'), None)
            if val_attr:
                const_val = numpy_helper.to_array(val_attr.t)
                is_scalar_const = const_val.size == 1

                if is_scalar_const:
                    current_lhs_symbol = f"{const_val.item():.4f}" # Use value as symbol
                    # No h_idx consumed for inlined scalar constants
                    equations.append(f"{current_lhs_symbol} (Represents Constant Value)")
                else:
                    current_lhs_symbol = f"h{h_idx}_const" # Tensor constant
                    h_idx += 1
                    equations.append(f"{current_lhs_symbol} = ConstantTensor(name: {node.name or output_onnx_name}, shape: {const_val.shape})")
                symbolic_name_map[output_onnx_name] = current_lhs_symbol
            else: # Constant might have value in other attributes or be an empty constant
                current_lhs_symbol = f"h{h_idx}_const_empty"
                h_idx+=1
                symbolic_name_map[output_onnx_name] = current_lhs_symbol
                equations.append(f"{current_lhs_symbol} = EmptyConstant(name: {node.name or output_onnx_name})")


        else: # Default handling for other ops
            current_lhs_symbol = f"h{h_idx}"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            h_idx += 1
            inputs_str = ", ".join(eq_rhs_input_symbols)
            equations.append(f"{current_lhs_symbol} = {op_type}({inputs_str})  # Generic op")

    # Print all collected equations
    print(f"--- Equations for {onnx_model_path} ---")
    if not equations:
        print("No equations generated. Model might be empty or contain only unsupported operations.")
    else:
        for eq in equations:
            # MODIFICATION HERE: Print the equation followed by two newlines for spacing
            print(f"{eq}\n\n")

    print("\n--- Graph Output Tensors ---")
    for out_info in graph.output:
        final_symbolic_name = symbolic_name_map.get(out_info.name, f"UNRESOLVED_OUTPUT({out_info.name})")
        print(f"{out_info.name} maps to: {final_symbolic_name}")

    print("\n-- Completed --")

if __name__ == '__main__':

    onnx_file_path_input = input(f"Enter the path to your ONNX file: ").strip()
    generate_equations(onnx_file_path_input)