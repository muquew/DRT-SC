import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=lambda item: sorted(item) if isinstance(item, (set, frozenset))
        else item.to_dict()).encode()).hexdigest()
