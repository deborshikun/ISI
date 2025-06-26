import onnx
from onnx import numpy_helper, shape_inference
import numpy as np
import os
import sys
import re
from z3 import *

def get_tensor_shape(graph, tensor_name_onnx, value_infos_map):
    """Helper to get the shape of an ONNX tensor."""
    if tensor_name_onnx in value_infos_map:
        info = value_infos_map[tensor_name_onnx]
        if info.type.tensor_type.HasField("shape"):
            shape = [dim.dim_value for dim in info.type.tensor_type.shape.dim if dim.HasField("dim_value") and dim.dim_value > 0]
            return shape
    return None

def generate_equations_to_file(onnx_model_path: str, output_file_path: str):
    """
    Parses an ONNX model, writes the Z3-friendly equations to a file,
    and also returns the equations and variable maps for immediate use.
    """
    print(f"[*] Stage 1: Parsing ONNX model '{onnx_model_path}'...")
    
    try:
        model = onnx.load(onnx_model_path)
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"Error loading or inferring shape of ONNX model: {e}")
        return None, None, None

    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    value_infos_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}
    
    equations = []
    symbolic_name_map = {}
    x_idx, h_idx = 1, 1

    for inp_info in graph.input:
        if inp_info.name not in initializers:
            symbolic_name_map[inp_info.name] = f"x{x_idx}"
            x_idx += 1
            
    # Process nodes in their given (topological) order
    for node in graph.node:
        op_type = node.op_type
        inputs_onnx = node.input
        outputs_onnx = node.output
        if not outputs_onnx: continue
        
        output_onnx_name = outputs_onnx[0]
        eq_rhs_input_symbols = [symbolic_name_map.get(name, name) for name in inputs_onnx]
        current_lhs_symbol = ""
        
        # --- Operator Handling (Restored to robust version) ---
        if op_type == 'MatMul':
            current_lhs_symbol = f"h{h_idx}_w"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            A_sym, W_name = eq_rhs_input_symbols[0], inputs_onnx[1]
            if W_name not in initializers: continue
            
            _W_matrix = initializers[W_name].T
            out_features, _ = _W_matrix.shape
            A_shape = get_tensor_shape(graph, inputs_onnx[0], value_infos_map)
            is_A_vector = A_shape and len(A_shape) > 0 and A_shape[-1] > 1

            for j in range(out_features):
                terms = [f"{weight_val:.4f}*{f'{A_sym}[{i}]' if is_A_vector else A_sym}" for i, weight_val in enumerate(_W_matrix[j, :]) if abs(weight_val) >= 1e-9]
                eq_rhs = " + ".join(terms) or "0"
                lhs_indexed = f"{current_lhs_symbol}[{j}]" if out_features > 1 else current_lhs_symbol
                equations.append(f"{lhs_indexed} = {eq_rhs}")

        elif op_type in ('Add', 'Sub'):
            input_A_sym = eq_rhs_input_symbols[0]
            if op_type == 'Add' and input_A_sym.endswith("_w"):
                base_name = input_A_sym[:-2]
                current_lhs_symbol = base_name
                h_idx += 1 # Consume the hidden layer index now
            else:
                current_lhs_symbol = f"h{h_idx}"; h_idx += 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol

            op_char = '+' if op_type == 'Add' else '-'
            op_out_shape = get_tensor_shape(graph, output_onnx_name, value_infos_map)
            num_elements = op_out_shape[-1] if op_out_shape and len(op_out_shape) > 0 else 1
            is_op_out_vector = num_elements > 1
            
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
                        is_scalar_like = not shape or shape[-1] == 1
                        terms.append(sym if is_scalar_like or not is_op_out_vector else f"{sym}[{j_op}]")
                lhs = f"{current_lhs_symbol}[{j_op}]" if is_op_out_vector else current_lhs_symbol
                equations.append(f"{lhs} = {f' {op_char} '.join(terms)}")

        elif op_type == 'Relu':
            input_symbol = eq_rhs_input_symbols[0]
            if input_symbol.endswith(("_w", "_a")) :
                base_name = input_symbol[:-2]
            else:
                base_name = input_symbol
            current_lhs_symbol = f"{base_name}_a"
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            
            shape = get_tensor_shape(graph, inputs_onnx[0], value_infos_map)
            num_elements = shape[-1] if shape else 1
            is_vector = num_elements > 1
            
            for i in range(num_elements):
                lhs = f"{current_lhs_symbol}[{i}]" if is_vector else current_lhs_symbol
                rhs = f"{input_symbol}[{i}]" if is_vector else input_symbol
                equations.append(f"{lhs} = Ite({rhs} > 0, {rhs}, 0)")

        elif op_type in ('Flatten', 'Reshape', 'Identity'):
            input_sym = eq_rhs_input_symbols[0]
            # This is a value-preserving operation for Z3. Pass the name through.
            symbolic_name_map[output_onnx_name] = input_sym

        else: # Generic fallback for any other operator
            current_lhs_symbol = f"h{h_idx}"; h_idx += 1
            symbolic_name_map[output_onnx_name] = current_lhs_symbol
            inputs_str = ", ".join(eq_rhs_input_symbols)
            equations.append(f"{current_lhs_symbol} = {op_type}({inputs_str}) # Generic op")

    # --- Write the formatted equations to the text file ---
    output_lines = [f"--- Equations for {os.path.basename(onnx_model_path)} ---"]
    output_blocks, current_block, last_base_name = [], [], ""
    def get_base_name(eq_str): return eq_str.split("=")[0].strip().split('[')[0].split('_')[0]
    
    for eq in equations:
        base_name = get_base_name(eq)
        if last_base_name and base_name != last_base_name:
            output_blocks.append("\n".join(current_block))
            current_block = []
        current_block.append(eq)
        last_base_name = base_name
    if current_block: output_blocks.append("\n".join(current_block))
    
    output_lines.append("\n\n".join(output_blocks))
    output_lines.append("\n\n--- Graph Output Tensors ---")
    for out_info in graph.output:
        final_name = symbolic_name_map.get(out_info.name, f"UNRESOLVED({out_info.name})")
        output_lines.append(f"{out_info.name} maps to: {final_name}")
    
    final_output_string = "\n".join(output_lines)
    try:
        with open(output_file_path, 'w') as f: f.write(final_output_string)
        print(f"✅ Equations saved to: {output_file_path}")
    except Exception as e:
        print(f"❌ Error saving equation file: {e}")
        
    return equations, symbolic_name_map, graph

def solve_equations_with_z3(equations: list, symbolic_name_map: dict, graph: onnx.GraphProto):
    """Builds and solves a Z3 problem from the given equations."""
    print("\n[*] Stage 2: Building Z3 problem...")

    var_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]')
    var_sizes = {}
    scalar_vars = set()

    for eq in equations:
        for var, index_str in var_pattern.findall(eq):
            index = int(index_str)
            var_sizes[var] = max(var_sizes.get(var, 0), index + 1)
        
        # Find scalar variables (those not indexed)
        # A simple way is to find names that don't have '[' after them
        all_names_in_eq = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', eq)
        for name in all_names_in_eq:
            if name not in var_sizes and name not in ['Ite', 'If']:
                scalar_vars.add(name)

    z3_vars = {}
    print("[*] Declaring Z3 variables...")
    for name, size in sorted(var_sizes.items()):
        z3_vars[name] = RealVector(name, size)
    for name in sorted(list(scalar_vars)):
         z3_vars[name] = Real(name)

    z3_context = z3_vars.copy()
    z3_context['If'] = If
    
    Equalities, RELU_Constraints = [], []
    print("[*] Converting equation strings to Z3 constraints...")
    for eq_str in equations:
        z3_eq_str = eq_str.replace('=', '==', 1)
        try:
            constraint = eval(z3_eq_str, {}, z3_context)
            if "If(" in z3_eq_str:
                 RELU_Constraints.append(constraint)
            else:
                Equalities.append(constraint)
        except Exception as e:
            print(f"Could not evaluate expression: '{z3_eq_str}'\nError: {e}")

    # --- >>> EDIT THIS SECTION FOR YOUR VERIFICATION PROBLEM <<< ---
    print("[!] PSA: Define your specific constraints in the 'Bounds' list.")
    Bounds = [
        # Example 1: Check if the output can ever be negative
        # z3_vars['h15'][0] < 0 
    ]

    # # --- Set up and run the solver ---
    # print("\n[*] Stage 3: Solving the constraint problem with Z3...")
    # s = Solver()
    # s.add(Equalities + RELU_Constraints)
    
    # if Bounds:
    #     print(f"[*] Adding {len(Bounds)} custom bound constraint(s)...")
    #     s.add(Bounds)
    
    # result = s.check()

    # print("\n--- Z3 Solver Result ---")
    # if result == sat:
    #     m = s.model()
    #     print("✅ SATISFIABLE")
    #     print("   A valid solution that satisfies all constraints was found.")
    # else:
    #     print(f"❌ {str(result).upper()}")
    #     print("   The solver could not find a solution for the given constraints.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        onnx_path = sys.argv[1]
    else:
        onnx_path = input("Enter the path to your ONNX file: ").strip()

    if not os.path.exists(onnx_path):
        print(f"FATAL: The file '{onnx_path}' was not found.")
        sys.exit(1)

    # Automatically determine the output filename for the equations
    output_eqns_path = os.path.splitext(onnx_path)[0] + "_equations.txt"
    
    # --- Main Pipeline ---
    equations_list, sym_map, onnx_graph = generate_equations_to_file(onnx_path, output_eqns_path)
    
    if equations_list:
        solve_equations_with_z3(equations_list, sym_map, onnx_graph)