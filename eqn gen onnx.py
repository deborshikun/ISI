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
                    shape.append('?')
            return shape
    return ['?']

def get_num_elements(shape):
    if not shape or any(s == '?' for s in shape):
        return 1
    return int(np.prod(shape))

def analyze_model(onnx_model_path: str):
    if not os.path.exists(onnx_model_path):
        print(f"Error: The file '{onnx_model_path}' was not found.")
        return None
    try:
        model = onnx.load(onnx_model_path)
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Error loading or processing ONNX model '{onnx_model_path}': {e}")
        return None

    graph = model.graph

    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    value_infos_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
    
    equations, symbolic_name_map, variable_bounds_info = [], {}, {}
    x_idx, h_idx = 1, 1

    for inp_info in graph.input:                                    #trying to get input names here
        if inp_info.name not in initializers:
            sym_name = f"x{x_idx}"
            symbolic_name_map[inp_info.name] = sym_name
            variable_bounds_info[sym_name] = 'Input'
            x_idx += 1

    producer_map = {out: node for node in graph.node for out in node.output}
    direct_activations_info = {}
    for node in graph.node:
        if node.op_type in ('Relu', 'Sigmoid', 'Tanh') and node.input:
            input_to_act_onnx = node.input[0]
            if input_to_act_onnx in producer_map and producer_map[input_to_act_onnx].op_type == 'Gemm':  #only ocnsidering activations directly after Gemm --issue 1
                direct_activations_info[input_to_act_onnx] = (node.op_type, node.output[0])

    for node in graph.node:
        op_type, inputs_onnx, outputs_onnx = node.op_type, node.input, node.output
        if not outputs_onnx: 
            continue
        output_onnx_name = outputs_onnx[0]
        eq_rhs_symbols = [symbolic_name_map.get(name, name) for name in inputs_onnx]

        if op_type == 'Gemm':
            base_h_name, current_lhs_symbol = f"h{h_idx}", f"h{h_idx}_w"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            variable_bounds_info[current_lhs_symbol] = 'Linear'
            
            X_sym, W_name = eq_rhs_symbols[0], inputs_onnx[1]
            if W_name not in initializers: 
                continue
            
            B_name = inputs_onnx[2] if len(inputs_onnx) > 2 else None
            W, B = initializers[W_name], initializers.get(B_name)
            transB = next((attr.i for attr in node.attribute if attr.name == 'transB'), 1)
            alpha = next((attr.f for attr in node.attribute if attr.name == 'alpha'), 1.0)
            beta = next((attr.f for attr in node.attribute if attr.name == 'beta'), 1.0)
            _W_matrix = W if transB else W.T
            out_features, in_features = _W_matrix.shape[0], _W_matrix.shape[1]
            
            for j in range(out_features):
                # Gemm treat as 1D vector
                terms = [f"{alpha*weight_val:.4f}*{X_sym}[{i}]" for i, weight_val in enumerate(_W_matrix[j, :in_features]) if abs(weight_val * alpha) >= 1e-9]
                eq_rhs = " + ".join(terms) or "0"
                if B is not None and abs(beta) > 1e-9 and abs(B.flatten()[j] * beta) >= 1e-9:
                    eq_rhs += f" + {B.flatten()[j] * beta:.4f}"
                equations.append(f"{current_lhs_symbol}[{j}] = {eq_rhs},")
                # if j == out_features - 1:
                #     equations[-1] = equations[-1].rstrip(',')  # last eqn doesnt have , 

            if output_onnx_name in direct_activations_info:
                act_op, act_out_name = direct_activations_info[output_onnx_name]
                post_act_lhs = f"{base_h_name}_a"
                symbolic_name_map[act_out_name] = post_act_lhs
                variable_bounds_info[post_act_lhs] = act_op
                for j in range(out_features):
                    lhs, rhs = f"{post_act_lhs}[{j}]", f"{current_lhs_symbol}[{j}]"
                    if act_op == 'Relu': 
                        equations.append(f"{lhs} = Ite({rhs} > 0, {rhs}, 0),")
                    elif act_op == 'Sigmoid': equations.append(f"{lhs} = 1 / (1 + Exp(-{rhs}))")
                    elif act_op == 'Tanh': equations.append(f"{lhs} = (Exp({rhs}) - Exp(-{rhs})) / (Exp({rhs}) + Exp(-{rhs}))")
                    if j == out_features - 1:
                        equations[-1] = equations[-1].rstrip(',') # last eqn doesnt have ,
            h_idx += 1
        
        elif op_type in ('Relu', 'Sigmoid', 'Tanh'):

            input_symbol = eq_rhs_symbols[0]
            base_name = input_symbol[:-2] if input_symbol.endswith("_w") else f"h{h_idx}"
            if not input_symbol.endswith("_w"): h_idx += 1
            current_lhs = f"{base_name}_a"
            symbolic_name_map[output_onnx_name] = current_lhs
            variable_bounds_info[current_lhs] = op_type
            
            num_elements = get_num_elements(get_tensor_shape(graph, inputs_onnx[0], value_infos_map))
            is_vector = num_elements > 1

            # for j in range(num_elements):
            #     lhs = f"{current_lhs}[{j}]" if is_vector else current_lhs
            #     rhs = f"{input_symbol}[{j}]" if is_vector else input_symbol
            #     if op_type == 'Relu':                                                                                                        #still working on this
            #         equations.append(f"{lhs} = Ite({rhs} > 0, {rhs}, 0)")
            #     elif op_type == 'Sigmoid': equations.append(f"{lhs} = 1 / (1 + Exp(-{rhs}))")
            #     elif op_type == 'Tanh': equations.append(f"{lhs} = (Exp({rhs}) - Exp(-{rhs})) / (Exp({rhs}) + Exp(-{rhs}))")

        elif op_type in ('Reshape', 'Flatten', 'Squeeze', 'Unsqueeze', 'Transpose', 'Identity'):

            # ensuring the graph remains connected and no "layers" are missed.
            input_sym = eq_rhs_symbols[0]
            current_lhs_symbol = f"h{h_idx}"
            h_idx += 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            variable_bounds_info[current_lhs_symbol] = variable_bounds_info.get(input_sym, 'Linear')
            
            # Create explicit element-wise identity equations for the flatten/reshape
            num_elements = get_num_elements(get_tensor_shape(graph, inputs_onnx[0], value_infos_map))
            for i in range(num_elements):
                equations.append(f"{current_lhs_symbol}[{i}] = {input_sym}[{i}]")
        
        else: # Generic fallback for other operators like Constant, Add, etc.
            current_lhs_symbol = f"h{h_idx}"
            h_idx += 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            variable_bounds_info[current_lhs_symbol] = op_type
            eq_rhs_str = f"{op_type}({', '.join(eq_rhs_symbols)})"
            equations.append(f"{current_lhs_symbol} = {eq_rhs_str}")

    equations[-1] = equations[-1].rstrip(',') # last equation doesn't have a trailing comma

    return {"equations": equations, "symbolic_name_map": symbolic_name_map, "variable_bounds_info": variable_bounds_info, "graph": graph,}

def format_and_write_output(analysis_results: dict, output_file_path: str, properties: dict):
    # This function remains the same as your version, it's correct.
    if not analysis_results: return
    equations, symbolic_name_map, var_bounds_info, graph = (
        analysis_results["equations"], analysis_results["symbolic_name_map"],
        analysis_results["variable_bounds_info"], analysis_results["graph"]
    )
    output_lines = []
    
    output_lines.append(f"--- Equations ---")
    output_lines.append("Equations[")
    if not equations: output_lines.append("")
    else:
        output_blocks, current_block, last_base_name = [], [], ""
        get_base_name = lambda eq_str: eq_str.split("=")[0].strip().split('[')[0].split('_')[0]
        for eq in equations:
            base_name = get_base_name(eq)
            if last_base_name and base_name != last_base_name: output_blocks.append("\n".join(current_block)); current_block = []
            current_block.append(f"  {eq}")
            last_base_name = base_name
        if current_block: output_blocks.append("\n".join(current_block))
        output_lines.append(",\n".join(output_blocks))
    output_lines.append("]")

    #output_lines.append("\n\n--- Bounds ---")
    prop_list_items = [f"  {var} {bound_str}" for var, bound_str in sorted(properties.items())]
    output_lines.append("\nProperties[")
    output_lines.append(",\n".join(prop_list_items))
    output_lines.append("]")

    bounds_list_items = []
    for var_base_name in sorted(var_bounds_info.keys()):
        if any(prop_var.startswith(var_base_name) for prop_var in properties): 
            continue
        op, bound_str = var_bounds_info[var_base_name], ""
        if op == 'Relu': bound_str = f"{var_base_name} in [0, inf)"
        elif op == 'Sigmoid': bound_str = f"{var_base_name} in [0, 1]"
        elif op == 'Tanh': bound_str = f"{var_base_name} in [-1, 1]"
        elif op in ('Linear', 'Input'): bound_str = f"{var_base_name} in (-inf, inf)"
        if bound_str: bounds_list_items.append(f"  {bound_str}")
    
    output_lines.append("\nBounds[")
    output_lines.append(",\n".join(bounds_list_items))
    output_lines.append("]")

    output_lines.append("\n\n Output Mapping")
    for out_info in graph.output:
        final_sym = symbolic_name_map.get(out_info.name, f"UNRESOLVED({out_info.name})")
        output_lines.append(f"{out_info.name} (Original Name) -> {final_sym}")

    with open(output_file_path, 'w') as f: f.write("\n".join(output_lines))
    print(f"\nOutput successfully saved to: {output_file_path}")

if __name__ == '__main__':
    onnx_file_path = input("Enter the path to your ONNX file: ").strip()
    analysis = analyze_model(onnx_file_path)

    if analysis:
        user_properties, graph, symbolic_map = {}, analysis["graph"], analysis["symbolic_name_map"]
        value_infos = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
        initializers = {init.name for init in graph.initializer}

        valid_symbols = set(symbolic_map.values())
        
        for section_name, io_list in [("Input", graph.input), ("Output", graph.output)]:
            print(f"\n{section_name} Properties\n")
            io_to_prompt = [(item.name, symbolic_map.get(item.name), get_tensor_shape(graph, item.name, value_infos))
                            for item in io_list if item.name not in initializers and symbolic_map.get(item.name)]
            
            if not io_to_prompt: print(f"No model {section_name.lower()}s found."); continue

            for onnx_name, sym_name, shape in io_to_prompt:
                while True:
                    print(f"\nProperties for {section_name.upper()}: '{sym_name}' (Original: {onnx_name}, Shape: {shape})")
                    print(f" Example: {sym_name}[i] > 0 (for all elements) OR {sym_name}[0] > 0 (for one element)")
                    prop_str = input("-> Enter property or type 'next' to continue: ").strip()

                    if prop_str.lower() == 'next': break
                    if not prop_str: 
                        continue
                    parts = prop_str.split(' ', 1)                   
                    var_name = parts[0].strip()
                    base_var = var_name.split('[')[0]
                    print(base_var, valid_symbols, sym_name)
                    if base_var not in sym_name:
                        print(f"'{base_var}' is not a recognized variable."); 
                        continue
                    if len(parts) == 2:               
                        user_properties[var_name] = parts[1].strip()
                        print(f"   Added: '{var_name} {parts[1].strip()}'")
                    else:
                        print("Invalid format.")

        output_filepath = os.path.splitext(onnx_file_path)[0] + "_equations.txt"
        format_and_write_output(analysis, output_filepath, user_properties)