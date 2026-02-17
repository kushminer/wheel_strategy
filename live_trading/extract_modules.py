#!/usr/bin/env python3
"""
Extract notebook code into Python modules.
This script parses the notebook and creates organized module files.
"""
import json
import re
from pathlib import Path

def extract_notebook_code(notebook_path):
    """Extract all code cells from notebook."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.strip() and not source.strip().startswith('#'):
                code_cells.append(source)
    
    return code_cells

def find_class_definitions(code):
    """Find all class definitions in code."""
    classes = {}
    pattern = r'^class\s+(\w+).*?:'
    for line in code.split('\n'):
        match = re.match(pattern, line)
        if match:
            class_name = match.group(1)
            classes[class_name] = line
    return classes

if __name__ == '__main__':
    nb_path = '3_csp_strategy_phase3.ipynb'
    cells = extract_notebook_code(nb_path)
    print(f"Found {len(cells)} code cells")
    
    # Analyze structure
    all_code = '\n'.join(cells)
    classes = find_class_definitions(all_code)
    print(f"\nFound classes: {', '.join(classes.keys())}")
