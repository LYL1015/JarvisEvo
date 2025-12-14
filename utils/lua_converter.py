import re

class LuaConverter:
    @staticmethod
    def _close_find(content, key_char, start=0):
        result = -1
        offset = start
        pair_map = {
            '\'': '\'',
            '"': '"',
            '[': ']',
            '{': '}',
        }
        pair_temp = []
        char = ''

        while offset < len(content):
            char_prev = char
            char = content[offset]
            if not char.isspace():
                if len(pair_temp) > 0:
                    last_pair_char = pair_temp[-1]
                    if char == last_pair_char:
                        pair_temp.pop()
                    elif char_prev != '\\' and char == '{' and last_pair_char == '}':
                        pair_temp.append(pair_map[char])
                else:
                    if char_prev != '\\' and char in pair_map:
                        pair_temp.append(pair_map[char])
                    elif char == key_char:
                        result = offset
                        break
            offset += 1

        return result

    @staticmethod
    def from_lua(content: str, level=0):
        content = content.strip()
        if level == 0:
            notes = re.findall(r'--\s*?\[\[[\s\S]*?\]\]', content)
            for note in notes:
                content = content.replace(note, '')
            notes = re.findall(r'--.*', content)
            for note in notes:
                content = content.replace(note, '')

        if len(content) == 0:
            raise Exception('Couldn\' t analyze blank content.')
        elif (content.count('-') == 0 or (content.count('-') == 1 and content[0] == '-')) \
                and content.replace('-', '').isnumeric():
            return int(content)
        elif (content.count('-') == 0 or (content.count('-') == 1 and content[0] == '-')) \
                and content.count('.') <= 1 and content.replace('-', '').replace('.', '').isnumeric():
            return float(content)
        elif content[:2] == '0x' and content.replace('0x', '').isalnum():
            return int(content, 16)
        elif content == 'false':
            return False
        elif content == 'true':
            return True
        elif content == 'nil':
            return None
        elif content[:2] == '[[' and content[-2:] == ']]':
            return content[2:-2]
        elif (content[0] == '"' and content[-1] == '"') or (content[0] == '\'' and content[-1] == '\''):
            return content[1:-1]
        elif content[0] == '{' and content[-1] == '}':
            content = content[1:-1]
            level += 1

            table = []
            count = 0
            offset = 0
            is_list = True
            while offset != -1:
                count += 1
                offset_prev = offset
                offset = LuaConverter._close_find(content, ',', offset + 1)
                if offset_prev != 0:
                    offset_prev += 1
                if offset == -1:
                    item = content[offset_prev:].strip()
                else:
                    item = content[offset_prev:offset].strip()
                if item != '':
                    divider_offset = LuaConverter._close_find(item, '=')
                    if divider_offset == -1:
                        key = count
                    else:
                        is_list = False
                        key = item[0:divider_offset].strip()
                        if key[0] == '[' and key[-1] == ']':
                            key = LuaConverter.from_lua(key[1:-1], level=level)

                    value = item[divider_offset + 1:].strip()
                    value = LuaConverter.from_lua(value, level=level)

                    table.append((key, value))

            if is_list:
                return [v for _, v in table]
            else:
                result = {}
                for k, v in table:
                    if isinstance(k, list):
                        k = tuple(k)
                    result[k] = v
                return result
        else:
            return None

    @staticmethod
    def _process_lightroom_json(obj):
        """Process JSON object for Lightroom-specific requirements"""
        if not isinstance(obj, dict):
            return obj
        
        # Create a copy to avoid modifying the original
        json_obj = obj.copy()
        
        # Remove specific keys if they exist
        keys_to_remove = ["CameraProfile", "CameraProfileDigest", "GrainSeed"]
        for key in keys_to_remove:
            if key in json_obj:
                del json_obj[key]

      
        # Handle tone curves
        tones = ["ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue"]
        for tone in tones:
            if tone in json_obj:
                # Convert string keys to numeric keys for the tone curve
                if isinstance(json_obj[tone], dict):
                    try:
                        json_obj[tone] = {int(k): v for k, v in json_obj[tone].items()}
                        if len(json_obj[tone]) % 2 != 0:
                            del json_obj[tone]
                    except (ValueError, TypeError):
                        # If conversion fails, remove the tone curve
                        del json_obj[tone]
        
        # Handle tone curves in Look.Parameters
        if "Look" in json_obj and "Parameters" in json_obj["Look"]:
            for tone in tones:
                if tone in json_obj["Look"]["Parameters"]:
                    try:
                        json_obj["Look"]["Parameters"][tone] = {int(k): v for k, v in json_obj["Look"]["Parameters"][tone].items()}
                        if len(json_obj["Look"]["Parameters"][tone]) % 2 != 0:
                            del json_obj["Look"]["Parameters"][tone]
                    except (ValueError, TypeError):
                        # If conversion fails, remove the tone curve
                        del json_obj["Look"]["Parameters"][tone]

        return json_obj

    @staticmethod
    def to_lua(obj, indent='    ', level=0):
        # Apply Lightroom processing at the top level
        if level == 0 and isinstance(obj, dict):
            obj = LuaConverter._process_lightroom_json(obj)
        
        result = ''

        if isinstance(obj, bool):
            result = str(obj).lower()
        elif type(obj) in [int, float]:
            result = str(obj)
        elif isinstance(obj, str):
            result = '"%s"' % obj
        elif type(obj) in [list, tuple]:
            is_simple_list = True
            for i in obj:
                if type(i) not in [bool, str, int, float]:
                    is_simple_list = False
                    break

            if is_simple_list:
                for i in obj:
                    if result != '':
                        result += ', '
                    result += LuaConverter.to_lua(i, level=level)
                result = '{%s}' % result
            else:
                level += 1
                for i in obj:
                    if result != '':
                        result += ',\n'
                    result += (indent * level) + LuaConverter.to_lua(i, level=level)
                result = '{\n%s\n%s}' % (result, indent * (level - 1))
        elif isinstance(obj, dict):
            level += 1
            for k, v in obj.items():
                if result != '':
                    result += ',\n'
                result += (indent * level)
                if isinstance(k, str):
                    # Check if the key contains special characters or is not a valid identifier
                    if not k.isidentifier() or '-' in k or '.' in k:
                        result += '[%s] = ' % LuaConverter.to_lua(k, level=level)
                    else:
                        result += '%s = ' % k
                else:
                    result += '[%s] = ' % LuaConverter.to_lua(k, level=level)
                result += LuaConverter.to_lua(v, level=level)
            result = '{\n%s\n%s}' % (result, indent * (level - 1))
        else:
            result = 'nil'

        return result