import onnx
from onnx import numpy_helper, shape_inference
import numpy as np
import os

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

def generate_equations(onnx_model_path: str, output_file_path: str = None):
    # --- This is the main function you've been building ---
    # It now accepts an output_file_path

    # A list to collect all output strings
    output_lines = []

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
    value_infos_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
    equations = []
    symbolic_name_map: dict[str, str] = {}
    x_idx, h_idx = 1, 1

    for inp_info in graph.input:
        if inp_info.name not in initializers:
            symbolic_name_map[inp_info.name] = f"x{x_idx}"
            x_idx += 1

    producer_map = {out_name: node for node in graph.node for out_name in node.output}
    direct_activations_info = {}
    for node in graph.node:
        if node.op_type in ('Relu', 'Sigmoid', 'Tanh') and node.input:
            input_to_act_onnx = node.input[0]
            if input_to_act_onnx in producer_map and producer_map[input_to_act_onnx].op_type == 'Gemm':
                direct_activations_info[input_to_act_onnx] = (node.op_type, node.output[0])

    for node_idx, node in enumerate(graph.node):
        op_type, inputs_onnx, outputs_onnx = node.op_type, node.input, node.output
        if not outputs_onnx: continue
        output_onnx_name = outputs_onnx[0]
        eq_rhs_input_symbols = [symbolic_name_map.get(name, name) for name in inputs_onnx]
        current_lhs_symbol = ""

        if op_type == 'Gemm':
            base_h_name = f"h{h_idx}"
            current_lhs_symbol = f"{base_h_name}_w"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            X_sym, W_name = eq_rhs_input_symbols[0], inputs_onnx[1]
            B_name = inputs_onnx[2] if len(inputs_onnx) > 2 else None
            if W_name not in initializers:
                equations.append(f"{current_lhs_symbol} = Error_Weight_Not_Found({W_name})")
                h_idx +=1
                continue
            W, B = initializers[W_name], initializers.get(B_name)
            transB = next((attr.i for attr in node.attribute if attr.name == 'transB'), 1)
            alpha = next((attr.f for attr in node.attribute if attr.name == 'alpha'), 1.0)
            beta = next((attr.f for attr in node.attribute if attr.name == 'beta'), 1.0)
            _W_matrix = W if transB else W.T
            out_features, in_features_from_weight = _W_matrix.shape
            X_shape = get_tensor_shape(graph, inputs_onnx[0], value_infos_map)
            is_X_vector_like = X_shape and ( (len(X_shape) > 1 and X_shape[1] > 1) or (len(X_shape) == 1 and X_shape[0] > 1) )
            for j in range(out_features):
                terms = [f"{alpha*weight_val:.4f}*{f'{X_sym}[{i}]' if is_X_vector_like else X_sym}" for i, weight_val in enumerate(_W_matrix[j, :]) if abs(weight_val * alpha) >= 1e-9]
                eq_rhs = " + ".join(terms) or "0"
                if B is not None and abs(beta) > 1e-9 and abs(B.flatten()[j] * beta) >= 1e-9:
                    eq_rhs += f" + {B.flatten()[j] * beta:.4f}"
                equations.append(f"{current_lhs_symbol}[{j}] = {eq_rhs}")
            if output_onnx_name in direct_activations_info:
                act_op_type, act_output_onnx_name = direct_activations_info[output_onnx_name]
                post_act_lhs_symbol = f"{base_h_name}_a"
                symbolic_name_map[act_output_onnx_name] = post_act_lhs_symbol
                gemm_out_shape = get_tensor_shape(graph, output_onnx_name, value_infos_map)
                is_gemm_out_vector = gemm_out_shape and gemm_out_shape[-1] > 1
                num_elements_act = gemm_out_shape[-1] if is_gemm_out_vector else 1
                for j_act in range(num_elements_act):
                    lhs = f"{post_act_lhs_symbol}[{j_act}]" if is_gemm_out_vector else post_act_lhs_symbol
                    rhs = f"{current_lhs_symbol}[{j_act}]" if is_gemm_out_vector else current_lhs_symbol
                    if act_op_type == 'Relu': equations.append(f"{lhs} = Ite({rhs} > 0, {rhs}, 0)")
                    elif act_op_type == 'Sigmoid': equations.append(f"{lhs} = 1 / (1 + Exp(-{rhs}))")
                    elif act_op_type == 'Tanh': equations.append(f"{lhs} = (Exp({rhs}) - Exp(-{rhs})) / (Exp({rhs}) + Exp(-{rhs}))")
                    else: equations.append(f"{lhs} = {act_op_type}({rhs})")
            h_idx += 1
        elif op_type == 'MatMul':
            current_lhs_symbol, A_sym, W_name = f"h{h_idx}_w", eq_rhs_input_symbols[0], inputs_onnx[1]
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            if W_name not in initializers:
                equations.append(f"{current_lhs_symbol} = Error_Weight_Not_Found({W_name})")
                h_idx += 1; continue
            _W_matrix = initializers[W_name].T
            out_features, in_features_from_weight = _W_matrix.shape
            A_shape = get_tensor_shape(graph, inputs_onnx[0], value_infos_map)
            is_A_vector_like = A_shape and ( (len(A_shape) > 1 and A_shape[1] > 1) or (len(A_shape) == 1 and A_shape[0] > 1) )
            for j in range(out_features):
                terms = [f"{weight_val:.4f}*{f'{A_sym}[{i}]' if is_A_vector_like else A_sym}" for i, weight_val in enumerate(_W_matrix[j, :]) if abs(weight_val) >= 1e-9]
                equations.append(f"{current_lhs_symbol}[{j}] = {' + '.join(terms) or '0'}")
        elif op_type in ('Add', 'Mul', 'Sub', 'Div'):
            input_A_sym = eq_rhs_input_symbols[0]
            if op_type == 'Add' and input_A_sym.endswith("_w"):
                current_lhs_symbol, h_idx = input_A_sym[:-2], h_idx + 1
            else:
                current_lhs_symbol, h_idx = f"h{h_idx}", h_idx + 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            op_out_shape = get_tensor_shape(graph, output_onnx_name, value_infos_map)
            is_op_out_vector = op_out_shape and ( (len(op_out_shape) > 1 and op_out_shape[-1] > 1) or (len(op_out_shape) == 1 and op_out_shape[0] > 1) )
            num_elements = op_out_shape[-1] if is_op_out_vector and len(op_out_shape) > 1 else (op_out_shape[0] if is_op_out_vector else 1)
            op_char = {'Add': '+', 'Mul': '*', 'Sub': '-', 'Div': '/'}[op_type]
            for j_op in range(num_elements):
                terms = []
                for i, onnx_name in enumerate(inputs_onnx):
                    if onnx_name in initializers:
                        flat_vals = initializers[onnx_name].flatten()
                        val = flat_vals[0] if flat_vals.size == 1 else flat_vals[j_op]
                        terms.append(f"{val:.4f}")
                    else:
                        sym = eq_rhs_input_symbols[i]
                        shape = get_tensor_shape(graph, onnx_name, value_infos_map)
                        is_scalar = not shape or shape == [1] or (len(shape) > 1 and shape[-1] == 1)
                        terms.append(sym if is_scalar or not is_op_out_vector else f"{sym}[{j_op}]")
                lhs = f"{current_lhs_symbol}[{j_op}]" if is_op_out_vector else current_lhs_symbol
                equations.append(f"{lhs} = {f' {op_char} '.join(terms)}")
        elif op_type in ('Relu', 'Sigmoid', 'Tanh'):
            if output_onnx_name in symbolic_name_map: continue
            input_symbol = eq_rhs_input_symbols[0]
            base_name = input_symbol[:-2] if input_symbol.endswith("_w") else f"h{h_idx}"
            if not input_symbol.endswith("_w"): h_idx += 1
            current_lhs_symbol = f"{base_name}_a"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            act_input_shape = get_tensor_shape(graph, inputs_onnx[0], value_infos_map)
            is_act_input_vector = act_input_shape and ( (len(act_input_shape) > 1 and act_input_shape[-1] > 1) or (len(act_input_shape) == 1 and act_input_shape[0] > 1) )
            num_elements_act = act_input_shape[-1] if is_act_input_vector and len(act_input_shape) > 1 else (act_input_shape[0] if is_act_input_vector else 1)
            for j_act in range(num_elements_act):
                lhs = f"{current_lhs_symbol}[{j_act}]" if is_act_input_vector else current_lhs_symbol
                rhs = f"{input_symbol}[{j_act}]" if is_act_input_vector and "[" not in input_symbol else input_symbol
                if op_type == 'Relu': equations.append(f"{lhs} = Ite({rhs} > 0, {rhs}, 0)")
                elif op_type == 'Sigmoid': equations.append(f"{lhs} = 1 / (1 + Exp(-{rhs}))")
                elif op_type == 'Tanh': equations.append(f"{lhs} = (Exp({rhs}) - Exp(-{rhs})) / (Exp({rhs}) + Exp(-{rhs}))")
        elif op_type in ('Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose', 'Identity'):
            input_sym, current_lhs_symbol, h_idx = eq_rhs_input_symbols[0], f"h{h_idx}", h_idx + 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            equations.append(f"{current_lhs_symbol} = {input_sym}") # {op_type} operation
        else:
            current_lhs_symbol, h_idx = f"h{h_idx}", h_idx + 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            equations.append(f"{current_lhs_symbol} = {op_type}({', '.join(eq_rhs_input_symbols)})") # Generic op

    # --- Build the final output ---
    output_lines.append(f"--- Equations for {onnx_model_path} ---")
    if not equations:
        output_lines.append("No equations generated. Model might be empty or contain only unsupported operations.")
    else:
        output_blocks = []
        current_block = []
        last_base_name = ""
        def get_base_name(eq_str):
            return eq_str.split("=")[0].strip().split('[')[0].split('_')[0]
        for eq in equations:
            base_name = get_base_name(eq)
            if last_base_name and base_name != last_base_name:
                output_blocks.append("\n".join(current_block))
                current_block = []
            current_block.append(eq)
            last_base_name = base_name
        if current_block:
            output_blocks.append("\n".join(current_block))
        
        # Add a single vertical space between each block of equations
        output_lines.append("\n\n".join(output_blocks))

    output_lines.append("\n\n--- Graph Output Tensors ---")
    for out_info in graph.output:
        final_symbolic_name = symbolic_name_map.get(out_info.name, f"UNRESOLVED_OUTPUT({out_info.name})")
        output_lines.append(f"{out_info.name} (Original Graph Output Name) maps to: {final_symbolic_name}")

    # Join all collected lines into a single string
    final_output_string = "\n".join(output_lines)
    
    # Print to console
    print(final_output_string)

    # Write to file if a path is provided
    if output_file_path:
        try:
            with open(output_file_path, 'w') as f:
                f.write(final_output_string)
            print(f"\nOutput successfully saved to: {output_file_path}")
        except Exception as e:
            print(f"\nError saving output to file: {e}")


if __name__ == '__main__':
    onnx_file_path_input = input(f"Enter the path to your ONNX file: ").strip()
    
    output_filename = os.path.splitext(onnx_file_path_input)[0] + "_equations.txt"
    generate_equations(onnx_file_path_input, output_filename)