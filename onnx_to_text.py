import onnx
from onnx import numpy_helper, shape_inference
import numpy as np
import os
import re

def get_tensor_shape(graph, tensor_name_onnx, value_infos_map):

    if tensor_name_onnx in value_infos_map:
        info = value_infos_map[tensor_name_onnx]
        if info.type.tensor_type.HasField("shape"):
            shape = []
            for dim in info.type.tensor_type.shape.dim:
                if dim.HasField("dim_value") and dim.dim_value > 0:
                    shape.append(dim.dim_value)
                else:
                    shape.append('?') # '?' for unknown or dynamic dimensions
            return shape
    return ['?']

def analyze_model(onnx_model_path: str):

    if not os.path.exists(onnx_model_path):
        print(f"Error: The file '{onnx_model_path}' was not found.")
        return None
    try:
        model = onnx.load(onnx_model_path)
    except Exception as e:
        print(f"Error loading ONNX model '{onnx_model_path}': {e}")
        return None

    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Warning: Error during shape inference for '{onnx_model_path}': {e}.")

    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    value_infos_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
    
    equations = []
    symbolic_name_map: dict[str, str] = {}
    variable_bounds_info: dict[str, str] = {}
    x_idx, h_idx = 1, 1

    for inp_info in graph.input:
        if inp_info.name not in initializers:
            sym_name = f"x{x_idx}"
            symbolic_name_map[inp_info.name] = sym_name
            variable_bounds_info[sym_name] = 'Input'
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
            variable_bounds_info[current_lhs_symbol] = 'Linear'
            
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
            out_features = _W_matrix.shape[0]
            X_shape = get_tensor_shape(graph, inputs_onnx[0], value_infos_map)
            is_X_vector_like = len(X_shape) > 0 and X_shape[-1] > 1
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
                variable_bounds_info[post_act_lhs_symbol] = act_op_type
                gemm_out_shape = get_tensor_shape(graph, output_onnx_name, value_infos_map)
                is_gemm_out_vector = len(gemm_out_shape) > 0 and gemm_out_shape[-1] > 1
                num_elements_act = gemm_out_shape[-1] if is_gemm_out_vector and '?' not in gemm_out_shape else 1
                for j_act in range(num_elements_act):
                    lhs = f"{post_act_lhs_symbol}[{j_act}]" if is_gemm_out_vector else post_act_lhs_symbol
                    rhs = f"{current_lhs_symbol}[{j_act}]" if is_gemm_out_vector else current_lhs_symbol
                    if act_op_type == 'Relu': equations.append(f"{lhs} = Ite({rhs} > 0, {rhs}, 0)")
                    elif act_op_type == 'Sigmoid': equations.append(f"{lhs} = 1 / (1 + Exp(-{rhs}))")
                    elif act_op_type == 'Tanh': equations.append(f"{lhs} = (Exp({rhs}) - Exp(-{rhs})) / (Exp({rhs}) + Exp(-{rhs}))")
                    else: equations.append(f"{lhs} = {act_op_type}({rhs})")
            h_idx += 1
        
        elif op_type in ('Relu', 'Sigmoid', 'Tanh'):
            if output_onnx_name in symbolic_name_map: continue
            input_symbol = eq_rhs_input_symbols[0]
            base_name = input_symbol[:-2] if input_symbol.endswith("_w") else f"h{h_idx}"
            if not input_symbol.endswith("_w"): h_idx += 1
            current_lhs_symbol = f"{base_name}_a"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            variable_bounds_info[current_lhs_symbol] = op_type
            act_input_shape = get_tensor_shape(graph, inputs_onnx[0], value_infos_map)
            is_act_input_vector = len(act_input_shape) > 0 and act_input_shape[-1] > 1
            num_elements_act = act_input_shape[-1] if is_act_input_vector and '?' not in act_input_shape else 1
            for j_act in range(num_elements_act):
                lhs = f"{current_lhs_symbol}[{j_act}]" if is_act_input_vector else current_lhs_symbol
                rhs = f"{input_symbol}[{j_act}]" if is_act_input_vector and "[" not in input_symbol else input_symbol
                if op_type == 'Relu': equations.append(f"{lhs} = Ite({rhs} > 0, {rhs}, 0)")
                elif op_type == 'Sigmoid': equations.append(f"{lhs} = 1 / (1 + Exp(-{rhs}))")
                elif op_type == 'Tanh': equations.append(f"{lhs} = (Exp({rhs}) - Exp(-{rhs})) / (Exp({rhs}) + Exp(-{rhs}))")

        elif op_type in ('Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose', 'Identity'):
            input_sym = eq_rhs_input_symbols[0]
            if re.match(r"h\d+.*", input_sym):
                 current_lhs_symbol = input_sym
            else:
                 current_lhs_symbol = f"h{h_idx}"
                 h_idx += 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            input_bound_type = variable_bounds_info.get(input_sym, 'Linear')
            variable_bounds_info[current_lhs_symbol] = input_bound_type
            # pass the name along for this 
        else:
            current_lhs_symbol = f"h{h_idx}"
            h_idx += 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            variable_bounds_info[current_lhs_symbol] = op_type
            equations.append(f"{current_lhs_symbol} = {op_type}({', '.join(eq_rhs_input_symbols)})")

    return {
        "equations": equations,
        "symbolic_name_map": symbolic_name_map,
        "variable_bounds_info": variable_bounds_info,
        "graph": graph,
    }

def format_and_write_output(analysis_results: dict, output_file_path: str, properties: dict):
    if not analysis_results: return

    equations, symbolic_name_map, var_bounds_info, graph = (
        analysis_results["equations"], analysis_results["symbolic_name_map"],
        analysis_results["variable_bounds_info"], analysis_results["graph"]
    )
    output_lines = []
    
    # Equations
    output_lines.append(f"--- Equations ---")
    if not equations:
        output_lines.append("No equations generated.")
    else:
        # Group equations by hidden layer for readability
        output_blocks, current_block, last_base_name = [], [], ""
        get_base_name = lambda eq_str: eq_str.split("=")[0].strip().split('[')[0].split('_')[0]
        for eq in equations:
            base_name = get_base_name(eq)
            if last_base_name and base_name != last_base_name:
                output_blocks.append("\n".join(current_block))
                current_block = []
            current_block.append(eq)
            last_base_name = base_name
        if current_block: output_blocks.append("\n".join(current_block))
        output_lines.append("\n\n".join(output_blocks))

    # Bounds (Combined)
    output_lines.append("\n\n--- Bounds ---")

    prop_list_items = [f"  {var} {bound_str}" for var, bound_str in sorted(properties.items())]
    output_lines.append("Properties[")
    output_lines.append(",\n".join(prop_list_items))
    output_lines.append("]")

    bounds_list_items = []
    sorted_vars = sorted(var_bounds_info.keys())
    for var_base_name in sorted_vars:
        if any(prop_var.startswith(var_base_name) for prop_var in properties):
            continue
        op, bound_str = var_bounds_info[var_base_name], ""
        if op == 'Relu': bound_str = f"{var_base_name} in [0, inf)"
        elif op == 'Sigmoid': bound_str = f"{var_base_name} in [0, 1]"
        elif op == 'Tanh': bound_str = f"{var_base_name} in [-1, 1]"
        elif op in ('Linear', 'Input', 'Gemm', 'Add', 'Mul'): bound_str = f"{var_base_name} in (-inf, inf)"
        else: continue
        bounds_list_items.append(f"  {bound_str}")
    
    output_lines.append("\nBounds[")
    output_lines.append(",\n".join(bounds_list_items))
    output_lines.append("]")

    # Graph Outputs
    output_lines.append("\n\n--- Graph Output Mapping ---")
    for out_info in graph.output:
        final_sym = symbolic_name_map.get(out_info.name, f"UNRESOLVED({out_info.name})")
        output_lines.append(f"{out_info.name} (Original Name) -> {final_sym}")

    # Final Print and Write
    final_output_string = "\n".join(output_lines)
    try:
        with open(output_file_path, 'w') as f: f.write(final_output_string)
        print(f"Output successfully saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving output to file: {e}")


if __name__ == '__main__':
    onnx_file_path = input("Enter the path to your ONNX file: ").strip()
    
    # analyzing the onnx
    analysis = analyze_model(onnx_file_path)

    if analysis:
        user_properties = {}
        graph, symbolic_map, value_infos, initializers = (
            analysis["graph"], analysis["symbolic_name_map"],
            {vi.name: vi for vi in list(analysis["graph"].value_info) + list(analysis["graph"].input) + list(analysis["graph"].output)},
            {init.name for init in analysis["graph"].initializer}
        )

        print("\n Input Properties \n")
        inputs_to_prompt = [
            (inp.name, symbolic_map.get(inp.name), get_tensor_shape(graph, inp.name, value_infos))
            for inp in graph.input if inp.name not in initializers and symbolic_map.get(inp.name)
        ]
        
        if not inputs_to_prompt: print("No model inputs found to define properties for.")
        else:
            for onnx_name, sym_name, shape in inputs_to_prompt:
                while True:
                    print(f"\nProperties for INPUT: '{sym_name}' (Original: {onnx_name}, Shape: {shape})")
                    print(f" {sym_name} in [,] (for whole tensor) OR {sym_name}[i] > x (for one element)")
                    prompt = "-> Enter property or type 'next' to continue: "
                    prop_str = input(prompt).strip()
                    if prop_str.lower() == 'next': break
                    if prop_str:
                        valid_names = set(symbolic_map.values())
                        # print(sym_name)
                        # print(valid_names)
                        parts = prop_str.split(' ', 1)
                        var_name = parts[0].strip()
                        if var_name not in sym_name:
                            print(f"'{parts[0].strip()}' is not a recognized variable.")
                            continue
                        if len(parts) == 2:
                            user_properties[parts[0].strip()] = parts[1].strip()
                            print(f"   Added: '{parts[0].strip()} {parts[1].strip()}'")
                        else: print("   Invalid format. Please use 'variable_name <condition>', e.g., 'x1 > 0'.")
        
        print("\n Output Properties \n")
        outputs_to_prompt = [
            (out.name, symbolic_map.get(out.name, "N/A"), get_tensor_shape(graph, out.name, value_infos))
            for out in graph.output
        ]
    
        if not outputs_to_prompt: print("No model outputs found to define properties for.")
        else:
            for onnx_name, sym_name, shape in outputs_to_prompt:
                while True:
                    print(f"\nProperties for OUTPUT: '{sym_name}' (Original: {onnx_name}, Shape: {shape})")
                    print(f" {sym_name} in [,] (for whole tensor) OR {sym_name}[i] > x (for one element)")
                    prompt = "-> Enter property or type 'next' to continue: "
                    prop_str = input(prompt).strip()
                    if prop_str.lower() == 'next': break
                    if prop_str:
                        # valid_names = set(symbolic_map.values())
                        # print(sym_name)
                        # print(valid_names)
                        parts = prop_str.split(' ', 1)
                        var_name = parts[0].strip()
                        if var_name not in sym_name:
                            print(f"'{parts[0].strip()}' is not a recognized variable.")
                            continue 
                        if len(parts) == 2:
                            user_properties[parts[0].strip()] = parts[1].strip()
                            print(f"   Added: '{parts[0].strip()} {parts[1].strip()}'")
                        else: print("   Invalid format. Please use 'variable_name <condition>'.")

        output_filepath = os.path.splitext(onnx_file_path)[0] + "_equations.txt"
        format_and_write_output(analysis, output_filepath, user_properties)