from enum import Enum
from collections import defaultdict
import torch
import torch.nn.functional as F


class Constants(Enum):
    IMAGE_RESOLUTION = 512
    MAX_ATTEN_RESOLUTION = 64

    TARGET_CROSS_RESOLUTION = 16
    TARGET_SELF_RESOLUTION = 32


class AttentionController(object):

    def __init__(self):

        self._self_attn = defaultdict(list)
        self._cross_attn = defaultdict(list)

        self._T = 0
        self._temp_T = 0

        self._layers_self = 0
        self._layers_cross = 0

        super().__init__()

    def set_attn_data(self, data, cls_tkn_pos, heads, attn_type=None):
        assert attn_type in ["self", "cross"], "the attention type must be specified!"

        data = self.up_sample(data, heads, cls_tkn_pos)

        self.layer_counter(attn_type)

        if attn_type == 'self':
            self._self_attn[self._layers_self].append(data)
        else:
            self._cross_attn[self._layers_cross].append(data)

    def layer_counter(self, attn_type=None):
        assert attn_type in ["self", "cross"], "the attention type must be specified!"

        if self._T != self._temp_T:
            if attn_type == 'self':
                self._layers_self += 1
            else:
                self._layers_cross += 1



    def up_sample(self, data, heads, cls_tkn_pos: int):
        result = self.pre_prosses(data, heads, cls_tkn_pos+1)
        return F.interpolate(result, size=(Constants.IMAGE_RESOLUTION.value, Constants.IMAGE_RESOLUTION.value), mode='bilinear', align_corners=False)


    def aggregate(self):
        L = len(self._cross_attn)
        T = self._T

        intermediate_tensor = torch.empty((Constants.IMAGE_RESOLUTION.value, Constants.IMAGE_RESOLUTION.value),
                                          dtype=self._cross_attn[0][0].dtype, device=self._cross_attn[0][0].device)
        maps = torch.zeros_like(intermediate_tensor)
        for _, times in self._cross_attn:
            maps += torch.stack(times).sum(dim=0)

        return maps

    def pre_prosses(self, data, heads, cls_tkn_pos):
        if not isinstance(cls_tkn_pos, list):
            result = data.view(-1, heads, data.shape[-2], data.shape[-1]).transpose(2, 3)[1, :, cls_tkn_pos, :].mean(dim=0)
            return result.view(Constants.TARGET_CROSS_RESOLUTION.value, Constants.TARGET_CROSS_RESOLUTION)
        else:
            raise NotImplementedError

    def increment_T(self):
        self._T += 1

        self._layers_self = 0
        self._layers_cross = 0


    def increment_temp_T(self):
        self._temp_T += 1
