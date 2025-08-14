import os
import pandas as pd
import pytest
import importlib.util

def load_parser_module(parser_name):
    """Dynamically load parser module"""
    spec = importlib.util.spec_from_file_location(
        f"{parser_name}_parser", 
        f"custom_parsers/{parser_name}_parser.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

@pytest.mark.parametrize("parser_name", ["icici"])
def test_parser(parser_name):
    """Test if parser correctly parses PDF and generates expected CSV"""
    # Skip test if parser doesn't exist
    parser_path = f"custom_parsers/{parser_name}_parser.py"
    if not os.path.exists(parser_path):
        pytest.skip(f"Parser {parser_name} not found")
    
    # Load parser module
    parser_module = load_parser_module(parser_name)
    
    # Define file paths
    pdf_path = f"data/{parser_name}/{parser_name}_sample.pdf"
    csv_path = f"data/{parser_name}/result.csv"
    
    # Skip test if sample files don't exist
    if not os.path.exists(pdf_path):
        pytest.skip(f"Sample PDF for {parser_name} not found")
    
    if not os.path.exists(csv_path):
        pytest.skip(f"Sample CSV for {parser_name} not found")
    
    # Parse PDF
    result_df = parser_module.parse(pdf_path)
    
    # Read expected CSV
    expected_df = pd.read_csv(csv_path)
    
    # Check if columns match
    assert list(result_df.columns) == list(expected_df.columns), \
        f"Column mismatch. Expected: {list(expected_df.columns)}, Got: {list(result_df.columns)}"
    
    # Check if data matches
    assert result_df.equals(expected_df), "Data mismatch between parsed result and expected CSV"