#!/usr/bin/env python3
"""Execute a Jupyter notebook and display outputs in real-time."""

import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

class OutputDisplayExecutor(ExecutePreprocessor):
    """Custom executor that displays outputs as they're generated."""

    def preprocess_cell(self, cell, resources, cell_index):
        """Execute a cell and display its output immediately."""
        # Execute the cell
        cell, resources = super().preprocess_cell(cell, resources, cell_index)

        # Display outputs immediately after execution
        if cell.cell_type == 'code' and cell.outputs:
            for output in cell.outputs:
                if output.output_type == 'stream':
                    print(output.text, end='', flush=True)
                elif output.output_type == 'execute_result':
                    if 'text/plain' in output.data:
                        print(output.data['text/plain'], flush=True)
                elif output.output_type == 'error':
                    print(f"ERROR: {output.ename}: {output.evalue}", flush=True)

        return cell, resources

def run_notebook(notebook_path):
    """Execute notebook and print outputs to terminal."""
    print(f"Executing notebook: {notebook_path}")
    print("=" * 60, flush=True)

    # Load the notebook
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Configure the preprocessor
    ep = OutputDisplayExecutor(timeout=600, kernel_name='python3')

    # Execute with output capture
    try:
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})

        # Save the executed notebook
        with open(notebook_path, 'w') as f:
            nbformat.write(nb, f)

        print("\n" + "=" * 60)
        print(f"✓ Notebook executed successfully: {notebook_path}")

    except Exception as e:
        print(f"\n✗ Error executing notebook: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python run_notebook.py <notebook_path>")
        sys.exit(1)

    run_notebook(sys.argv[1])
