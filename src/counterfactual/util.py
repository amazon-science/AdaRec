"""
Utility functions for counterfactual generation.
Copied and adapted from Potemkin.
"""

def format_value(value, data_type):
    """
    Format a value according to its data type for display in profiles.
    
    Args:
        value: The value to format
        data_type (str): The type of the value ('binary', 'percent', 'integer', 'float', 'string')
    
    Returns:
        str: Formatted value as string
    """
    if value is None or (isinstance(value, float) and value != value):  # Handle NaN
        return "N/A"
    
    if data_type == 'binary':
        return 'Yes' if value else 'No'
    elif data_type == 'percent':
        return f"{value:.1%}"
    elif data_type == 'integer':
        return f"{int(value):,}"
    elif data_type == 'float':
        return f"{float(value):.2f}"
    else:
        return str(value)
