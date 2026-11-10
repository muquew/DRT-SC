import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile


def _invalid_constant(value):
    raise ValueError(f'non-finite JSON number: {value}')


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f'duplicate JSON key: {key}')
        value[key] = item
    return value


def _json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('JSON number exceeds the finite range')
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
        parse_constant=_invalid_constant, parse_float=_json_float,
        object_pairs_hook=_unique_pairs)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    temporary = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                prefix='.'+path.name, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
