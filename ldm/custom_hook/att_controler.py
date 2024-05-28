from enum import Enum


class Constants(Enum):
    IMAGE_RESOLUTION = 512
    MAX_ATTEN_RESOLUTION = 64


class AttentionController(object):

    def __init__(self):

        self._self_attn = list
        self._cross_attn = list

        super().__init__()

    def set_attn_data(self, data, attn_type=None, resolution=None):
        assert attn_type in ["self", "cross"], "the attention type must be specified!"

        if attn_type == 'self':
            self._self_attn.append({resolution: data})
        else:
            self._cross_attn.append({resolution: data})

    def up_sample(self):
        pass

    def aggregate(self):
        def normalize():
            pass
        pass