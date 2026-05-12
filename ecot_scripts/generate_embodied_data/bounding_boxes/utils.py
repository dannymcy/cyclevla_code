import json

import numpy as np


class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def post_process_caption(caption, lang_instruction):
    text = caption.replace(",", ".")
    if text[-1] != ".":
        text += "."
    return text
