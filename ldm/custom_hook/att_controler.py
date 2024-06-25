from enum import Enum
from collections import defaultdict
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import numpy as np
import os
from panopticapi.utils import IdGenerator
import json
from collections import defaultdict


class Constants(Enum):
    IMAGE_RESOLUTION = 512
    MAX_ATTEN_RESOLUTION = 64

    TARGET_CROSS_RESOLUTION = 16
    TARGET_SELF_RESOLUTION = 32

    ALPHA = 0.7
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

        # self._image_counter = len(os.listdir("/content/masks"))-1

        self.panoptic_dict = defaultdict(list)
        self.grounding_dict = defaultdict(list)

        self.annots_idx = 0


        self.get_categories_info()


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

        # self.create_dataset(masks)

        # return _cross, _self, self.post_process(_cross, _self)
        return masks


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


    def create_dataset(self, masks, label_folder:str, image_name:int):

        cats = {category['id']: category for category in self.cats}
        generator = IdGenerator(cats)

        label = " ".join(label_folder.strip().split("_"))

        mask = masks[self.token_idx, :, :]
        _max_value = torch.max(mask.flatten())
        _min_value = torch.min(mask.flatten())
        mask = (mask - _min_value) / (_max_value - _min_value)

        mask_bg = mask < Constants.ALPHA.value
        mask_u = (mask >= Constants.ALPHA.value) & (mask < Constants.BETA.value)
        otherwise = ~(mask_bg | mask_u)
        mask[mask_bg] = 0
        mask[mask_u] = 255
        mask[otherwise] = 0

        _, binary = cv2.threshold(mask.cpu().numpy().astype(np.uint8), 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.panoptic_dict["images"].append({"file_name": f"{image_name:05}.jpg", "height": 512, "width": 512, "id": int(image_name)})
        self.grounding_dict["images"].append({"file_name": f"{image_name:05}.jpg", "height": 512, "width": 512, "id": int(image_name)})

        the_annot_of_panoptic_gt = {"segments_info": [], "file_name": f"{image_name:05}.png",
                                    "image_id": int(image_name)}

        raw_mask = np.zeros((512, 512, 3), dtype=np.uint8)

        # filling the image with background color.
        bg_info = list(filter(lambda cat: cat["name"] == "void", self.cats))[0]
        _id_gb, color_bg = generator.get_id_and_color(cat_id=int(bg_info["id"]))
        gb_color = (int(color_bg[2]), int(color_bg[1]), int(color_bg[0]))
        raw_mask[:, :] = gb_color



        background_segment_info = {"id": _id_gb, "category_id": bg_info["id"], "iscrowd": 0, "bbox": [0, 0, 512, 512], "area": 0}
        the_annot_of_panoptic_gt['segments_info'].append(background_segment_info)


        for contour in contours:
            segment_info = {}

            if len(np.array(contour).flatten()) <= 7:
                continue

            polygon_np = np.array(contour, np.int32)
            poly = polygon_np.reshape((-1, 1, 2))

            x, y, w, h = cv2.boundingRect(polygon_np)
            bbox = [x, y, w, h]
            area = cv2.contourArea(poly)

            the_category = list(filter(lambda cat: cat["name"] == label, self.cats))[0]

            # grounding
            _poly = contour.flatten().tolist()

            a_grounding = {"segmentation": [_poly], "area": int(area), "iscrowd": int(0),
                           "image_id": int(image_name), "category_id": int(the_category["id"]),
                           "bbox": bbox, "id": int(self.annots_idx), "split": "train", "ann_id": int(self.annots_idx)}

            prompt = label
            _sentences = list()
            tokens = prompt.split(' ')
            raw = sent = prompt
            _sentences.append({"tokens": tokens, "raw": raw, "sent_id": int(0), "sent": sent})
            a_grounding.update({"sentences": _sentences})
            self.grounding_dict["annotations"].append(a_grounding)

            # panoptic
            _rgb2id, color = generator.get_id_and_color(cat_id=int(the_category["id"]))
            segment_info["id"] = _rgb2id
            segment_info["category_id"] = int(the_category["id"])
            segment_info["iscrowd"] = 0
            segment_info["bbox"] = bbox
            segment_info["area"] = int(area)

            the_annot_of_panoptic_gt["segments_info"].append(segment_info)
            cv2.fillPoly(raw_mask, [contour], color=(int(color[2]), int(color[1]), int(color[0])))

            self.annots_idx += 1

        # this one
        # name = f'/content/_dataset/{label_folder}/masks/{image_name:05}.png'


        # cv2.imwrite(name, mask.cpu().numpy().astype(np.uint8))
        self.panoptic_dict["annotations"].append(the_annot_of_panoptic_gt)
        cv2.imwrite(os.path.join('results', 'panoptic', f'{image_name:05}.png'), raw_mask)



    def get_categories_info(self):

        # with open("/content/_meta_data/CATEGORIES.json", 'r') as file:
        with open(os.path.join('_meta_data', 'CATEGORIES.json'), 'r') as file:
            cats = json.load(file)

        self.cats = cats

        # self.cats = {category['id']: category for category in cats}


        # filtered = masks[self.token_idx, :, :]
        # for idx in range(filtered.shape[0]):
        #     mask = filtered[idx, :, :]
        #
        #     _max_value = torch.max(mask.flatten())
        #     _min_value = torch.min(mask.flatten())
        #     mask = (mask - _min_value) / (_max_value - _min_value)
        #
        #     mask_bg = mask < Constants.ALPHA.value
        #     mask_u = (mask >= Constants.ALPHA.value) & (mask < Constants.BETA.value)
        #     otherwise = ~(mask_bg | mask_u)
        #     mask[mask_bg] = 0
        #     mask[mask_u] = 255
        #     mask[otherwise] = 0
        #
        #     output_dir = '/content/masks/'
        #     name = f'{output_dir}{self._image_counter:05}_{self.lbl}.png'
        #
        #     img = mask.cpu().numpy().astype(np.uint8)
        #     image = Image.fromarray(img)
        #     image.save(name)
        #     self._image_counter += 1
        #
        #
        #     # cv2.imwrite(name, mask)

    def save_annots(self):
        # panoptic_path = "/content/masks/panoptic.json"
        panoptic_path = os.path.join('results', 'panoptic.json')
        with open(panoptic_path, 'w') as file:
            json.dump(self.panoptic_dict, file)

        # grounding_path = "/content/masks/grounding.json"
        grounding_path = os.path.join('results', 'grounding.json')
        with open(grounding_path, 'w') as file:
            json.dump(self.grounding_dict, file)




