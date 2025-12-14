import re
import random
import string
import os
from lua_exchange import LuaExchange

def generate_random_hash(length=5):
    """Generate a random hash string of specified length"""
    return ''.join(random.choices(string.hexdigits, k=length)).upper()

def convert_curve_format(curve_data):
    """
    Convert curve data from lua format with indexed keys to lrtemplate format.
    
    Input format: [1] = 0, [2] = 19, ...
    Output format: 0, 19, ...
    """
    # Extract values using regex
    values = []
    pattern = r'\[\d+\]\s*=\s*(-?\d+(?:\.\d+)?)'
    matches = re.findall(pattern, curve_data)
    
    if not matches:
        return "{}"
    
    # Format as lrtemplate style
    formatted_values = []
    for value in matches:
        formatted_values.append(f"\t\t\t{value}")
    
    return "{\n" + ",\n".join(formatted_values) + ",\n\t\t}"
    
def process_lua_content(content):
    """Process lua content and extract/transform the required data"""
    # Find all curve data that needs conversion
    curve_keys = [
        'ToneCurvePV2012', 
        'ToneCurvePV2012Red', 
        'ToneCurvePV2012Green', 
        'ToneCurvePV2012Blue',
        'MainCurve',
        'RedCurve',
        'GreenCurve',
        'BlueCurve'
    ]
    
    # Extract each curve section and convert it
    for key in curve_keys:
        pattern = fr'{key}\s*=\s*\{{[^{{}}]*(\[\d+\][^{{}}]*)+\}}'
        match = re.search(pattern, content)
        if match:
            curve_section = match.group(0)
            curve_data = curve_section[curve_section.find('{')+1:curve_section.rfind('}')]
            converted_curve = convert_curve_format(curve_data)
            
            # Replace the original curve data with the converted format
            content = content.replace(curve_section, f"{key} = {converted_curve}")
    
    return content

def fix_indentation(content):
    """Fix the indentation of the content to match lrtemplate format"""
    # Split by lines
    lines = content.split('\n')
    indented_lines = []
    
    # Add proper indentation
    for line in lines:
        line = line.strip()
        if line:
            indented_lines.append(f"\t\t\t{line}")
    
    return '\n'.join(indented_lines)

def cleanup_content(content):
    """Clean up the content to remove extra curly braces and fix formatting"""
    # Remove the opening curly brace at the beginning if present
    content = re.sub(r'^\s*{\s*', '', content)
    
    # Remove the closing curly brace at the end if present
    content = re.sub(r'\s*}\s*$', '', content)
    
    return content

def reverse_map_temperature_tint(mapped_value, type='temperature'):
    # Create reverse mapping tables
    temp_map = {
        2000: -100, 2500: -71, 3000: -49, 3500: -33, 4000: -20, 4500: -9, 5000: 0,
        5500: 8, 6000: 14, 6500: 20, 7000: 25, 7500: 30, 8000: 34, 8500: 38, 9000: 41,
        9500: 44, 10000: 47, 10500: 50, 20000: 78, 30000: 89, 40000: 96, 50000: 100
    }
    tint_map = {
        -150: -100, -140: -96, -130: -92, -120: -88, -110: -83, -100: -78,
        -90: -74, -70: -63, -50: -51, -40: -45, -20: -30, 0: -11, 10: 1,
        30: 22, 50: 39, 70: 54, 80: 60, 90: 67, 100: 73, 120: 84, 140: 95,
        150: 100
    }
    
    # Select mapping table and create reverse mapping
    original_map = temp_map if type == 'temperature' else tint_map
    reverse_map = {v: k for k, v in original_map.items()}
    
    # If input value exists in reverse mapping, return directly
    if mapped_value in reverse_map:
        return reverse_map[mapped_value]
    
    # Find the two closest values for linear interpolation
    keys = sorted(reverse_map.keys())
    if mapped_value < keys[0]:
        return reverse_map[keys[0]]
    if mapped_value > keys[-1]:
        return reverse_map[keys[-1]]
    
    # Find the two closest values
    for i in range(len(keys)-1):
        if keys[i] <= mapped_value <= keys[i+1]:
            # Linear interpolation
            x1, x2 = keys[i], keys[i+1]
            y1, y2 = reverse_map[x1], reverse_map[x2]
            return y1 + (y2 - y1) * (mapped_value - x1) / (x2 - x1)

def transform_json_obj(json_obj):
    # Convert IncrementalTemperature and IncrementalTint to Temperature and Tint
    if "WhiteBalance" in json_obj and json_obj["WhiteBalance"] == "Custom":
        if "IncrementalTemperature" in json_obj:
            incre_temperature = json_obj["IncrementalTemperature"]
            json_obj["Temperature"] = reverse_map_temperature_tint(incre_temperature)
            del json_obj["IncrementalTemperature"]
        if "IncrementalTint" in json_obj:
            incre_tint = json_obj["IncrementalTint"]
            json_obj["Tint"] = reverse_map_temperature_tint(incre_tint, type="tint")
            del json_obj["IncrementalTint"]
    return json_obj

def lua_to_lrtemplate(lua_file_path, output_path=None):
    """Convert a Lightroom config.lua file to lrtemplate format"""
    if not output_path:
        # Generate output path based on input path if not provided
        base_name = os.path.splitext(os.path.basename(lua_file_path))[0]
        output_path = os.path.join(os.path.dirname(lua_file_path), f"{base_name}.lrtemplate")
    
    # Read the lua file
    with open(lua_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    

    # Remove the "return" at the beginning if present
    content = re.sub(r'^return\s*', '', content.strip())

    # transform incre-temperature and incre-tint to temperature and tint
    json_obj = LuaExchange.from_lua(content)
    transformed_json_obj = transform_json_obj(json_obj)
    content = LuaExchange.to_lua(transformed_json_obj)
    
    # Clean up the content
    content = cleanup_content(content)
    
    # Process the content to convert curve formats
    processed_content = process_lua_content(content)
    
    # Fix indentation for the entire content
    processed_content = fix_indentation(processed_content)
    
    # Generate random hash for internalName
    random_hash = generate_random_hash(5)
    preset_name = f"JarvisArt-{random_hash}"
    # Create the lrtemplate structure
    lrtemplate = f"""s = {{
	id = "",
	internalName = "Preset-{random_hash}",
	title = "{preset_name}",
	type = "Develop",
	value = {{
		settings = {{
{processed_content}
		}},
	}},
	version = 0,
}}
"""
    
    # Write the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(lrtemplate)
    
    return output_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python lua2lrt.py <input_lua_file> [output_lrtemplate_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    
    # If a second argument is provided, use it as the output file path
    # Otherwise, use the same path as input but with .lrtemplate extension
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    try:
        output_path = lua_to_lrtemplate(input_file, output_file)
        print(f"Conversion successful! Output saved to: {output_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)