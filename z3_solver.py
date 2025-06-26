import os
import re
import sys

def create_z3_script(input_equations_path: str):
    """
    Reads a file containing model equations and generates a runnable Z3 Python script
    that is structurally similar to the user's NNDemo.py example.

    Args:
        input_equations_path (str): The path to the .txt file with the equations.
    """
    if not os.path.exists(input_equations_path):
        print(f"Error: Input file not found at '{input_equations_path}'")
        return

    # --- Phase 1: Discover all variables and their sizes ---
    print(f" Phase 1: Discovering variables from '{input_equations_path}'...")
    var_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\[(\d+)\]')
    var_sizes = {}
    with open(input_equations_path, 'r') as f:
        for line in f:
            if line.strip().startswith('---') or line.strip().startswith('#'):
                continue
            matches = var_pattern.finditer(line)
            for match in matches:
                var_name, index = match.group(1), int(match.group(2))
                var_sizes[var_name] = max(var_sizes.get(var_name, 0), index + 1)

    print(f"Found {len(var_sizes)} vector variables.")

    # --- Phase 2: Parse all equations and categorize them ---
    print("✍️  Phase 2: Parsing and categorizing all equations...")
    equality_strings = []
    relu_strings = []

    with open(input_equations_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('---'):
                continue

            if '#' in line:
                if any(op in line for op in ['Flatten', 'Reshape', 'Transpose']):
                    line = line.replace("=", "==") # Keep the identity relation
                else:
                    line = line.split('#')[0].strip()

            parts = line.split('=', 1)
            if len(parts) != 2:
                continue

            z3_eq = f"{parts[0].strip()} == {parts[1].strip()}"

            if "Ite(" in z3_eq:
                relu_strings.append(z3_eq.replace("Ite(", "If(", 1))
            else:
                equality_strings.append(z3_eq)

    # --- Phase 3: Generate the Z3 script content in the desired format ---
    print("🏗️  Phase 3: Assembling the final Z3 script...")
    
    output_script_lines = []
    
    # Add imports and variable declarations
    output_script_lines.append("from z3 import *")
    output_script_lines.append("\n# --- Variable Declarations ---")
    for name, size in sorted(var_sizes.items()):
        output_script_lines.append(f"{name} = RealVector('{name}', {size})")

    # Create the Equalities list
    output_script_lines.append("\n# --- Equalities ---")
    output_script_lines.append("Eq = [")
    for eq in equality_strings:
        output_script_lines.append(f"    {eq},")
    output_script_lines.append("]\n")

    # Create the RELU_Constraints list
    output_script_lines.append("# --- RELU Constraints ---")
    output_script_lines.append("RELU = [")
    for eq in relu_strings:
        output_script_lines.append(f"    {eq},")
    output_script_lines.append("]\n")

    # Create the placeholder Bounds list
    output_script_lines.append("# --- Bounds ---")
    output_script_lines.append("# Placeholder for user-defined input/output constraints")
    output_script_lines.append("# For example:")
    output_script_lines.append("# Bounds = [")
    output_script_lines.append("#     x1[0] >= 0,")
    output_script_lines.append("#     x1[0] <= 1,")
    output_script_lines.append("#     h15[0] > 0")
    output_script_lines.append("# ]")
    output_script_lines.append("Bounds = []\n")

    # Add the solver command, exactly like NNDemo.py
    output_script_lines.append("# --- Solve ---")
    output_script_lines.append("solve(RELU + Bounds + Eq)")
    
    # --- Phase 4: Write the generated script to a file ---
    output_script_path = os.path.splitext(input_equations_path)[0] + "_z3_script.py"
    print(f"\n Phase 4: Writing final script to '{output_script_path}'...")
    
    try:
        with open(output_script_path, 'w') as f:
            f.write("\n".join(output_script_lines))
        print(f"Successfully generated Z3 script!")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = input("Enter the path to your '..._equations.txt' file: ").strip()
    
    create_z3_script(file_path)