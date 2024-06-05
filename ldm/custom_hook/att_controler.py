from enum import Enum
from collections import defaultdict
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import os


class Constants(Enum):
    IMAGE_RESOLUTION = 512
    MAX_ATTEN_RESOLUTION = 64

    TARGET_CROSS_RESOLUTION = 16
    TARGET_SELF_RESOLUTION = 32

    ALPHA = 0.75
    BETA = 1.01
    THAU = 0.75



class AttentionController(object):

    def __init__(self):

        self.lbl = None
        self.token_idx = -1

        self._self_attn = defaultdict(list)
        self._cross_attn = defaultdict(list)

        self._T = 0
        self._temp_T = 0

        self._layers_self = 0
        self._layers_cross = 0

        self._image_counter = len(os.listdir("/content/masks"))-1



        super().__init__()

    def reset_attn_data(self):
        self.lbl = None
        self.token_idx = -1

        self._self_attn = defaultdict(list)
        self._cross_attn = defaultdict(list)

        self._T = 0
        self._temp_T = 0
        self._layers_self = 0
        self._layers_cross = 0


    def set_attn_data(self, data, cls_tkn_pos, heads, attn_type=None):
        assert attn_type in ["self", "cross"], "the attention type must be specified!"

        # if attn_type == "cross":
        data = self.up_sample(data, heads, cls_tkn_pos, attn_type)

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

    def up_sample(self, data, heads, cls_tkn_pos: int, attn_type):
        return self.pre_processes(data, heads, cls_tkn_pos + 1, attn_type)
        # res =  F.interpolate(result,
        #                      size=(Constants.TARGET_SELF_RESOLUTION.value, Constants.TARGET_SELF_RESOLUTION.value),
        #                      mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

    def post_process(self, cross_attn, self_attn):

        final = (self_attn ** Constants.THAU.value) @ cross_attn
        f = final.transpose(0, 1).view(-1, 32, 32).unsqueeze(0)
        final = F.interpolate(f, size=(Constants.IMAGE_RESOLUTION.value, Constants.IMAGE_RESOLUTION.value),
                              mode='bilinear', align_corners=False).squeeze(0).view(1, -1).transpose(0, 1)

        final = final.transpose(0, 1).view(-1, 512, 512)
        return final

        # _max, _arg_max = torch.max(final, dim=-1)
        # _max = _max.view(Constants.IMAGE_RESOLUTION.value, Constants.IMAGE_RESOLUTION.value).cpu().numpy()
        # _arg_max = _arg_max.view(Constants.IMAGE_RESOLUTION.value, Constants.IMAGE_RESOLUTION.value).cpu().numpy()
        #
        # mask_bg = _max <= Constants.ALPHA.value
        # mask_u = (_max > Constants.ALPHA.value) & (_max < Constants.BETA.value)
        # otherwise = ~(mask_bg | mask_u)
        #
        # _max[mask_bg] = 0
        # _max[mask_u] = 255
        #
        # _max[otherwise] = _arg_max[otherwise]



        # # final = final.transpose(0, 1).view(-1, 512, 512)
        # final = final.view(512, 512, -1)[:, :, 0]
        #
        # final = (final - final.min()) / (final.max() - final.min())
        #
        # # _max, _arg_max = torch.max(final, dim=-1)
        #
        # mask_bg = final <= Constants.ALPHA.value
        # mask_u = (final > Constants.ALPHA.value) & (final < Constants.BETA.value)
        # other = ~( mask_bg| mask_u)
        #
        # final[mask_bg] = 0
        # final[mask_u] = 0
        # final[other] = 255

        return _max

    def aggregate(self):

        L_CROSS = len(self._cross_attn)
        L_SELF = len(self._self_attn)

        T = self._T

        maps_cross = torch.zeros_like(self._cross_attn[1][0])
        maps_self = torch.zeros_like(self._self_attn[1][0])

        for _, times in self._cross_attn.items():
            maps_cross += torch.stack(times).sum(dim=0)

        for _, times in self._self_attn.items():
            maps_self += torch.stack(times).sum(dim=0)

        # _cross, _self = self.post_proce()
        _cross = maps_cross / (T * L_CROSS)
        _self = maps_self / (T * L_SELF)
        masks = self.post_process(_cross, _self)

        self.create_dataset(masks)

        # return _cross, _self, self.post_process(_cross, _self)
        return 0, 0, 0


    def pre_processes(self, data, heads, cls_tkn_pos, attn_type):
        if attn_type == "cross":
            if not isinstance(cls_tkn_pos, list):
                result = data.view(-1, heads, data.shape[-2], data.shape[-1]).transpose(2, 3)[1, :, :, :].mean(dim=0).view(-1, 16, 16).unsqueeze(0)
                if attn_type == "cross":

                    return F.interpolate(result, size=(32, 32), mode='bilinear', align_corners=False).squeeze(0).view(77, -1).transpose(0, 1)

                    # return result.view(Constants.TARGET_CROSS_RESOLUTION.value, Constants.TARGET_CROSS_RESOLUTION.value).unsqueeze(0).unsqueeze(0)
                else:
                    return result.view(Constants.TARGET_SELF_RESOLUTION.value,
                                       Constants.TARGET_SELF_RESOLUTION.value).unsqueeze(0).unsqueeze(0)

            else:
                raise NotImplementedError
        else:
            return data.view(-1, heads, data.shape[-2], data.shape[-1])[1, :, :, :].mean(dim=0)

    def increment_T(self):
        self._T += 1

        self._layers_self = 0
        self._layers_cross = 0

    def increment_temp_T(self):
        self._temp_T += 1

    def set_token_idx(self, idx):
        self.token_idx = idx

    def set_token_lbl(self, lbl):
        self.lbl = lbl


    def create_dataset(self, masks):
        filtered = masks[self.token_idx, :, :]
        for idx in range(filtered.shape[0]):
            mask = filtered[idx, :, :]

            _max_value = torch.max(mask.flatten())
            _min_value = torch.min(mask.flatten())
            mask = (mask - _min_value) / (_max_value - _min_value)

            mask_bg = mask < Constants.ALPHA.value
            mask_u = (mask >= Constants.ALPHA.value) & (mask < Constants.BETA.value)
            otherwise = ~(mask_bg | mask_u)
            mask[mask_bg] = 0
            mask[mask_u] = 255
            mask[otherwise] = 0

            output_dir = '/content/masks/'
            name = f'{output_dir}{self._image_counter:05}_{self.lbl}.png'

            img = mask.cpu().numpy().astype(np.uint8)
            image = Image.fromarray(img)
            image.save(name)
            self._image_counter += 1


            # cv2.imwrite(name, mask)
            # self._image_counter += 1
            #



