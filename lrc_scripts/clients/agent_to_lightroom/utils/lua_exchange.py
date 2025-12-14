import re


class LuaExchange:
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
            notes = re.findall('--\s*?\[\[[\s\S]*?\]\]', content)
            for note in notes:
                content = content.replace(note, '')
            notes = re.findall('--.*', content)
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
                offset = LuaExchange._close_find(content, ',', offset + 1)
                if offset_prev != 0:
                    offset_prev += 1
                if offset == -1:
                    item = content[offset_prev:].strip()
                else:
                    item = content[offset_prev:offset].strip()
                if item != '':
                    divider_offset = LuaExchange._close_find(item, '=')
                    if divider_offset == -1:
                        key = count
                    else:
                        is_list = False
                        key = item[0:divider_offset].strip()
                        if key[0] == '[' and key[-1] == ']':
                            key = LuaExchange.from_lua(key[1:-1], level=level)

                    value = item[divider_offset + 1:].strip()
                    value = LuaExchange.from_lua(value, level=level)

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
    def to_lua(obj, indent='    ', level=0):
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
                    result += LuaExchange.to_lua(i, level=level)
                result = '{%s}' % result
            else:
                level += 1
                for i in obj:
                    if result != '':
                        result += ',\n'
                    result += (indent * level) + LuaExchange.to_lua(i, level=level)
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
                        result += '[%s] = ' % LuaExchange.to_lua(k, level=level)
                    else:
                        result += '%s = ' % k
                else:
                    result += '[%s] = ' % LuaExchange.to_lua(k, level=level)
                result += LuaExchange.to_lua(v, level=level)
            result = '{\n%s\n%s}' % (result, indent * (level - 1))
        else:
            result = 'nil'

        return result