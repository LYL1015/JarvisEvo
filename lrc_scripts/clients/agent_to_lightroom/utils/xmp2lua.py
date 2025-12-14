import xml.etree.ElementTree as ET

version_strings = ['Version', 'ProcessVersion']
crop_strings = ['CropTop', 'CropLeft', 'CropBottom', 'CropRight', 'CropAngle', 'CropConstrainToWarp', 'HasCrop']

exclude_keys = ['Version', 
                'ToneCurvePV2012', 'ToneCurvePV2012Red', 
                'ToneCurvePV2012Green', 'ToneCurvePV2012Blue',
                'CorrectionSyncID','MaskSyncID']

def parse_value(value):
    """Convert XMP value to Lua format"""
    if value.startswith('+'):
        value = value[1:]
        
    try:
        num = float(value)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        # Handle boolean values
        if value.lower() == 'true':
            return 'true'
        if value.lower() == 'false':
            return 'false'
        # Return string values with quotes
        return f'"{value}"'

def parse_tone_curve(seq_elem):
    """Parse global tone curve sequence elements into Lua table format"""
    result = []
    points = []
    for item in seq_elem.findall('rdf:li', namespaces={'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}):
        if item.text:
            x, y = map(str.strip, item.text.split(','))
            points.append((x, y))
    
    # Format points as Lua table entries
    i = 1
    for x, y in points:
        result.append(f"[{i}] = {x}")
        result.append(f"[{i+1}] = {y}")
        i += 2
    
    return result

def parse_local_tone_curve(seq_elem):
    """Parse local adjustment tone curve sequence elements into Lua table format"""
    result = []
    points = []
    for item in seq_elem.findall('rdf:li', namespaces={'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#"}):
        if item.text:
            points.append(f'"{item.text}"')
    
    # Format points as Lua table entries
    for i, point in enumerate(points, 1):
        result.append(f"[{i}] = {point}")
    
    return result

def print_element_tree(element, level=0):
    """Debug helper to print XML structure"""
    print("  " * level + element.tag)
    for child in element:
        print_element_tree(child, level + 1)

def parse_reference_point(value):
    """Parse reference point into Lua table format"""
    if value:
        x, y = map(float, value.split())
        return f"{{{x}, {y}}}"
    return "{}"

def parse_mask_dabs(dabs_elem, ns):
    """Parse mask dabs sequence into Lua table format"""
    result = []
    result.append("            Dabs = {")
    for dab in dabs_elem.findall('rdf:li', ns):
        if dab.text:
            result.append(f'              "{dab.text}",')
    result.append("            },")
    return result

def parse_point_models(points_elem, ns):
    """Parse point models sequence into Lua table format"""
    result = []
    result.append("          PointModels = {")
    for point in points_elem.findall('rdf:li', ns):
        if point.text:
            values = point.text
            result.append(f'            "{values}",')
    result.append("          },")
    return result

def parse_gesture(gesture_elem, ns):
    """Parse gesture element into Lua table format"""
    result = []
    result.append("        Gesture = {")
    
    # Handle each gesture item
    for gesture_item in gesture_elem.findall('rdf:Seq/rdf:li/rdf:Description', ns):
        result.append("          {")
        
        # Process gesture attributes
        for key, value in gesture_item.attrib.items():
            if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                clean_key = key.split('}')[1]
                if clean_key not in exclude_keys:
                    result.append(f"            {clean_key} = {parse_value(value)},")
        
        # Handle Points if they exist
        dabs_elem = gesture_item.find('crs:Points/rdf:Seq', ns)
        if dabs_elem is not None:
            result.append("            Points = {")
            for dab in dabs_elem.findall('rdf:li', ns):
                result.append("            {")
                for key, value in dab.attrib.items():
                    if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                        clean_key = key.split('}')[1]
                        result.append(f"              {clean_key} = {parse_value(value)},")
                result.append("            },")
            result.append("            },")
        result.append("          },")
    
    result.append("        },")
    return result

def parse_correction_masks(masks_elem, ns):
    """Parse correction masks into Lua table format"""
    result = []
    result.append("    CorrectionMasks = {")
    
    # Handle both types of mask structures
    for mask_item in masks_elem.findall('rdf:li', ns):
        if len(mask_item.attrib) > 0:  # Direct attributes on rdf:li
            result.append("      {")
            for key, value in mask_item.attrib.items():
                if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                    clean_key = key.split('}')[1]
                    if clean_key not in exclude_keys:
                        result.append(f"        {clean_key} = {parse_value(value)},")
            result.append("      },")
        else:  # Complex mask structure with nested Description
            for mask in mask_item.findall('rdf:Description', ns):
                result.append("      {")
                # Process mask attributes
                for key, value in mask.attrib.items():
                    if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                        clean_key = key.split('}')[1]
                        if clean_key not in exclude_keys:
                            result.append(f"        {clean_key} = {parse_value(value)},")
                
                # Handle nested Masks
                nested_masks = mask.find('crs:Masks/rdf:Seq', ns)
                if nested_masks is not None:
                    result.append("        Masks = {")
                    for nested_mask in nested_masks.findall('rdf:li/rdf:Description', ns):
                        result.append("          {")
                        for key, value in nested_mask.attrib.items():
                            if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                                clean_key = key.split('}')[1]
                                if clean_key not in exclude_keys:
                                    result.append(f"            {clean_key} = {parse_value(value)},")
                        
                        # Handle Dabs
                        dabs_elem = nested_mask.find('crs:Dabs/rdf:Seq', ns)
                        if dabs_elem is not None:
                            result.extend(parse_mask_dabs(dabs_elem, ns))
                        result.append("          },")
                    result.append("        },")
                
                # Handle CorrectionRangeMask(light range)
                range_mask = mask.find('crs:CorrectionRangeMask', ns)
                if range_mask is not None:
                    result.append("        CorrectionRangeMask = {")
                    for key, value in range_mask.attrib.items():
                        if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                            clean_key = key.split('}')[1]
                            if clean_key in ["Type", "Version", "SampleType"]:
                                # Handle number type values
                                result.append(f'          {clean_key} = {value},')
                            elif clean_key == "Invert":
                                # Handle boolean type values
                                result.append(f'          {clean_key} = {value.lower()},')
                            elif clean_key == "LumRange" or clean_key == "LuminanceDepthSampleInfo":
                                # Handle string type values
                                result.append(f'          {clean_key} = "{value}",')
                            else:
                                result.append(f'          {clean_key} = {parse_value(value)},')
                    
                    # Handle PointModels (color range)
                    description = range_mask.find('rdf:Description', ns)
                    if description is not None:
                        for key, value in description.attrib.items():
                            if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                                clean_key = key.split('}')[1]
                                result.append(f'          {clean_key} = {parse_value(value)},')
                    points_elem = range_mask.find('rdf:Description/crs:PointModels/rdf:Seq', ns)
                    if points_elem is not None:
                        result.extend(parse_point_models(points_elem, ns))
                    result.append("        },")

                # Handle Gesture
                gesture_elem = mask.find('crs:Gesture', ns)
                if gesture_elem is not None:
                    result.extend(parse_gesture(gesture_elem, ns))

                result.append("      },")
    
    result.append("    },")
    return result

def parse_single_tag_attributes(elem, tag_name, ns, indent=2):
    """Parse single tag with attributes into Lua table format"""
    result = []
    spaces = " " * indent
    result.append(f"{spaces}{tag_name} = {{")
    
    for key, value in elem.attrib.items():
        if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
            clean_key = key.split('}')[1]
            if clean_key == "FocalRange":
                value = "0 0 100 100"
            result.append(f"{spaces}  {clean_key} = {parse_value(value)},")
    
    result.append(f"{spaces}}},")
    return result

def parse_mask_group(mask_group, ns):
    """Parse mask group corrections into Lua table format"""
    result = []
    result.append("{")
    
    corrections = mask_group.findall('rdf:Seq/rdf:li/rdf:Description', ns)
    
    for i, correction in enumerate(corrections, 1):
        result.append(f"  {{")
        
        # Process correction attributes
        for key, value in correction.attrib.items():
            if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                clean_key = key.split('}')[1]
                if clean_key not in exclude_keys:
                    if clean_key in version_strings:
                        result.append(f"    {clean_key} = \"{value}\",")
                    else:
                        result.append(f"    {clean_key} = {parse_value(value)},")
        
        # Handle local adjustment curves
        for curve in ['MainCurve', 'RedCurve', 'GreenCurve', 'BlueCurve']:
            curve_elem = correction.find(f'crs:{curve}', ns)
            if curve_elem is not None:
                result.append(f"    {curve} = {{")
                seq = curve_elem.find('rdf:Seq', ns)
                if seq is not None:
                    result.extend(f"      {line}," for line in parse_local_tone_curve(seq))
                result.append("    },")
        
        # Handle LocalPointColors
        local_point_colors = correction.find('crs:LocalPointColors/rdf:Seq', ns)
        if local_point_colors is not None:
            result.append("    LocalPointColors = {")
            for i, point in enumerate(local_point_colors.findall('rdf:li', ns), 1):
                if point.text:
                    result.append(f'      [{i}] = "{point.text}",')
            result.append("    },")
        
        # Handle CorrectionMasks
        masks_elem = correction.find('crs:CorrectionMasks/rdf:Seq', ns)
        if masks_elem is not None:
            result.extend(parse_correction_masks(masks_elem, ns))
        result.append("  },")
    result.append("}")
    return result

def parse_look_parameters(params, ns):
    """Parse Look parameters into Lua table format"""
    result = []
    result.append("  Parameters = {")
    
    # Handle basic parameters
    for key, value in params.attrib.items():
        if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
            clean_key = key.split('}')[1]
            if clean_key in version_strings:
                result.append(f"    {clean_key} = \"{value}\",")
            else:
                result.append(f"    {clean_key} = {parse_value(value)},")
    
    # Handle tone curves in parameters
    for curve in ['ToneCurvePV2012', 'ToneCurvePV2012Red', 
                  'ToneCurvePV2012Green', 'ToneCurvePV2012Blue']:
        curve_elem = params.find(f'crs:{curve}', ns)
        if curve_elem is not None:
            result.append(f"    {curve} = {{")
            seq = curve_elem.find('rdf:Seq', ns)
            if seq is not None:
                result.extend(f"      {line}," for line in parse_tone_curve(seq))
            result.append("    },")
    
    result.append("  },")
    return result

def parse_alt_group(alt_elem, ns):
    """Parse Alt group elements into Lua table format"""
    result = []
    result.append("{")
    
    for li in alt_elem.findall('rdf:li', ns):
        lang = li.get('{http://www.w3.org/XML/1998/namespace}lang')
        if lang and li.text:
            result.append(f'["{lang}"] = "{li.text}",')
    
    result.append("}")
    return result

def parse_look(look_elem, ns):
    """Parse Look element into Lua table format"""
    result = []
    result.append("{")
    
    # Check if this is a single tag with attributes
    if len(look_elem) == 0:
        # Handle single tag case
        for key, value in look_elem.attrib.items():
            if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                clean_key = key.split('}')[1]
                if clean_key in version_strings:
                    result.append(f"  {clean_key} = \"{value}\",")
                else:
                    result.append(f"  {clean_key} = {parse_value(value)},")
    else:
        # Handle complex nested structure
        look_desc = look_elem.find('rdf:Description', ns)
        if look_desc is not None:
            # Handle basic attributes
            for key, value in look_desc.attrib.items():
                if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
                    clean_key = key.split('}')[1]
                    if clean_key in version_strings:
                        result.append(f"  {clean_key} = \"{value}\",")
                    else:
                        result.append(f"  {clean_key} = {parse_value(value)},")
            
            # Handle Group element inside Look
            group_elem = look_desc.find('crs:Group/rdf:Alt', ns)
            if group_elem is not None:
                result.append("  Group = ")
                result.extend("  " + line for line in parse_alt_group(group_elem, ns))
                result.append(",")
            
            # Handle Parameters
            params = look_desc.find('crs:Parameters', ns)
            if params and len(params.attrib.items()) == 0:
                params = look_desc.find('crs:Parameters/rdf:Description', ns)
            if params is not None:
                result.extend(parse_look_parameters(params, ns))
    
    result.append("}")
    return result

def parse_point_colors(point_colors_elem, ns):
    """Parse PointColors element into Lua table format"""
    result = []
    result.append("PointColors = {")
    
    # Find all point color entries
    for point in point_colors_elem.findall('rdf:Seq/rdf:li', ns):
        if point.text:
            # Split the values and remove any whitespace
            values = [float(x.strip()) for x in point.text.split(',')]
            
            # Create a point color entry
            result.append("  {")
            result.append(f"    SrcHue = {values[0]},")
            result.append(f"    SrcSat = {values[1]},")
            result.append(f"    SrcLum = {values[2]},")
            result.append(f"    HueShift = {values[3]},")
            result.append(f"    SatScale = {values[4]},")
            result.append(f"    LumScale = {values[5]},")
            result.append(f"    RangeAmount = {values[6]},")
            
            # Add HueRange
            result.append("    HueRange = {")
            result.append(f"      LowerNone = {values[7]},")
            result.append(f"      LowerFull = {values[8]},")
            result.append(f"      UpperFull = {values[9]},")
            result.append(f"      UpperNone = {values[10]},")
            result.append("    },")
            
            # Add SatRange
            result.append("    SatRange = {")
            result.append(f"      LowerNone = {values[11]},")
            result.append(f"      LowerFull = {values[12]},")
            result.append(f"      UpperFull = {values[13]},")
            result.append(f"      UpperNone = {values[14]},")
            result.append("    },")
            
            # Add LumRange
            result.append("    LumRange = {")
            result.append(f"      LowerNone = {values[15]},")
            result.append(f"      LowerFull = {values[16]},")
            result.append(f"      UpperFull = {values[17]},")
            result.append(f"      UpperNone = {values[18]},")
            result.append("    },")
            
            result.append("  },")
    
    result.append("},")
    return result

def parse_xmp(xmp_file):
    """Parse XMP file and convert to Lua table format"""
    tree = ET.parse(xmp_file)
    root = tree.getroot()
    
    ns = {
        'crs': "http://ns.adobe.com/camera-raw-settings/1.0/",
        'rdf': "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    }
    
    result = []
    result.append("{")  # Start of main table
    desc = root.find('.//rdf:Description', ns)
    
    # Process basic attributes
    for key, value in desc.attrib.items():
        if key.startswith('{http://ns.adobe.com/camera-raw-settings/1.0/}'):
            clean_key = key.split('}')[1]
            if clean_key not in exclude_keys:
                if "table_" in clean_key.lower() or clean_key in crop_strings: 
                    continue
                if clean_key in version_strings:
                    result.append(f"{clean_key} = \"{value}\",")
                else:
                    result.append(f"{clean_key} = {parse_value(value)},")
    
    
    # Handle tone curves
    for curve in ['ToneCurvePV2012', 'ToneCurvePV2012Red', 
                  'ToneCurvePV2012Green', 'ToneCurvePV2012Blue']:
        curve_elem = desc.find(f'crs:{curve}', ns)
        if curve_elem is not None:
            result.append(f"{curve} = {{")
            seq = curve_elem.find('rdf:Seq', ns)
            if seq is not None:
                result.extend(f"  {line}," for line in parse_tone_curve(seq))
            result.append("},")
    
    # Handle mask groups
    mask_group = desc.find('crs:MaskGroupBasedCorrections', ns)
    if mask_group is not None:
        result.append("MaskGroupBasedCorrections = ")
        result.extend(parse_mask_group(mask_group, ns))
        result.append(",")
    
    # Handle Look data
    look_elem = desc.find('crs:Look', ns)
    if look_elem is not None:
        result.append("Look = ")
        result.extend(parse_look(look_elem, ns))
        result.append(",")
    
    # Handle LensBlur
    lens_blur = desc.find('crs:LensBlur', ns)
    if lens_blur is not None:
        result.extend(parse_single_tag_attributes(lens_blur, "LensBlur", ns))

    # Handle DepthMapInfo
    depth_map = desc.find('crs:DepthMapInfo', ns)
    if depth_map is not None:
        result.extend(parse_single_tag_attributes(depth_map, "DepthMapInfo", ns))
    
    # Handle PointColors
    point_colors = desc.find('crs:PointColors', ns)
    if point_colors is not None:
        result.extend(parse_point_colors(point_colors, ns))
    
    result.append("}")  # End of main table
    return '\n'.join(result)

def main():
    # Parse XMP and save as Lua table
    lua_table = parse_xmp('config.xmp')
    with open('converted_settings.lua', 'w', encoding='utf-8') as f:
        f.write('return ' + lua_table)

if __name__ == '__main__':
    main()