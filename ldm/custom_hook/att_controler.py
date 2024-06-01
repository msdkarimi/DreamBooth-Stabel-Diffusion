from enum import Enum
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class CONSTANTS(Enum):
    IMAGE_RESOLUTION = 512
    MAX_ATTEN_RESOLUTION = 64
    TEXT_MAX_LEN = 77
    TARGET_CROSS_RESOLUTION = 64


class AttentionController(object):

    def __init__(self, total_steps, num_points=16, refine=True):

        self._self_attn = {64: [], 32: [], 16: [], 8: []}
        # self._cross_attn = {64: [], 32: [], 16: [], 8: []}
        self._cross_attn = defaultdict(list)

        self._T = 0
        self._temp_T = 0

        self.T = total_steps

        # Generate the  gird
        self.grid = self.generate_sampling_grid(num_points)
        # Inialize other parameters
        self.kl_threshold = np.array([0.9] * 3)
        self.refine = refine

        self._layers_self = 0
        self._layers_cross = 0

    def layer_counter(self, attn_type=None):
        assert attn_type in ["self", "cross"], "the attention type must be specified!"

        if self._T != self._temp_T:
            if attn_type == 'self':
                self._layers_self += 1
            else:
                self._layers_cross += 1

    def generate_sampling_grid(self, num_of_points):
        segment_len = 63 // (num_of_points - 1)
        total_len = segment_len * (num_of_points - 1)
        start_point = (63 - total_len) // 2
        x_new = np.linspace(start_point, total_len + start_point, num_of_points)
        y_new = np.linspace(start_point, total_len + start_point, num_of_points)
        x_new, y_new = np.meshgrid(x_new, y_new, indexing='ij')
        points = np.concatenate(([x_new.reshape(-1, 1), y_new.reshape(-1, 1)]), axis=-1).astype(int)
        return points

    def set_attn_data(self, attension, cls_tkn_pos, heads, position=None):
        assert position in ["down", "up", "middel"], "the attention type must be specified!"

        if self._temp_T == self.T:
            uc_c, spatial_dim, dim = attension.shape

            resolution = spatial_dim ** 0.5

            if (resolution == 64) and (spatial_dim != dim):
                result = attension.view(-1, heads, attension.shape[-2], attension.shape[-1]).transpose(2, 3)[1, :,
                         cls_tkn_pos + 1:cls_tkn_pos + 2, :].mean(dim=0).view(-1,
                                                                              CONSTANTS.TARGET_CROSS_RESOLUTION.value,
                                                                              CONSTANTS.TARGET_CROSS_RESOLUTION.value)
                self._cross_attn[self._layers_cross].append(result)

            if spatial_dim == dim:
                self._self_attn[resolution].append(attension.view(-1, heads, spatial_dim, spatial_dim))
            else:
                pass
                # self._cross_attn[resolution].append(attension.view(-1, heads, spatial_dim, dim))

    def aggregate_cross_attention(self, weight_ratio=[1.0, 1.0, 1.0, 1.0]):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aggregated_weights = torch.zeros((CONSTANTS.IMAGE_RESOLUTION.value,
                                         CONSTANTS.IMAGE_RESOLUTION.value,
                                         CONSTANTS.TEXT_MAX_LEN.value), device=device)

        if weight_ratio is None:
            weight_ratio = self.get_weight_ratio()

        for index, (_, _map) in enumerate(self._cross_attn.items()):
            _map = _map[1, :, :, :]
            h, spatial, dim = _map.shape
            size = int(spatial ** 0.5)
            scale_factor = int(CONSTANTS.IMAGE_RESOLUTION.value/size)

            _map = _map.mean(0).permute(0, 1)
            _map = _map.view(dim, size, size).unsqueeze(0)
            _map = F.interpolate(_map, scale_factor=scale_factor, mode='bilinear', align_corners=False).squeeze(0)
            _map = _map.permute(1, 2, 0)
            _map = _map / torch.sum(_map, dim=-1, keepdim=True)
            # weights = weights / tf.math.reduce_sum(weights, axis=-1, keepdims=True)

            aggregated_weights += _map*weight_ratio[index]

        return aggregated_weights.to(torch.float64).cpu().numpy()


    def preprocess_attention_maps(self):
        for resolution, weights in self._self_attn.items():
            self._self_attn[resolution] = torch.stack(weights).sum(dim=0) / len(weights)

        # for resolution, weights in self._cross_attn.items():
        #     self._cross_attn[resolution] = torch.stack(weights).sum(dim=0) / len(weights)

    def cross_process(self):
        L_CROSS = len(self._cross_attn)
        T = self._T
        maps_cross = torch.zeros_like(self._cross_attn[1][0])
        for _, times in self._cross_attn.items():
            maps_cross += torch.stack(times).sum(dim=0)
        maps_cross = maps_cross / (T * L_CROSS)
        maps_cross = maps_cross.unsqueeze(0)
        return F.interpolate(maps_cross,
                      size=(CONSTANTS.IMAGE_RESOLUTION.value, CONSTANTS.IMAGE_RESOLUTION.value),
                      mode='bilinear', align_corners=False).squeeze(0)

    def semantic_diffSeg(self):


        # aggregate self-attention/cross-attention maps in different layers
        self.preprocess_attention_maps()

        segments = self.segment()[0]

        # cross = self.aggregate_cross_attention()
        cross = self.cross_process().permute(1, 2, 0)

        return segments, cross

    def segment(self, weight_ratio=None):
        M_list = []
        for i in range(len(self._self_attn[64])):
            # Step 1: Attention Aggregation
            weights = self.aggregate_weights(i, weight_ratio=weight_ratio)
            # Step 2 & 3: Iterative Merging & NMS
            M_final = self.generate_masks(weights)
            M_list.append(M_final)
        return np.array(M_list)

    def mask_merge(self, iter, attns, kl_threshold, grid=None):
        device = torch.device('cpu')
        attns = torch.Tensor(attns, device=device)
        if iter == 0:
            # The first iteration of merging
            anchors = attns[grid[:, 0], grid[:, 1], :, :]  # 256 x 64 x 64
            anchors = torch.unsqueeze(anchors, dim=1)  # 256 x 1 x 64 x 64
            attns = attns.reshape(1, 4096, 64, 64)
            # 256 x 4096 x 64 x 64 is too large for a single gpu, splitting into 16 portions
            split = np.sqrt(grid.shape[0]).astype(int)
            kl_bin = []
            for i in range(split):
                anchors_casted = anchors[i * split:(i + 1) * split].to(torch.float16)
                attns_casted = attns.to(torch.float16)
                temp = self.KL(anchors_casted, attns_casted) < kl_threshold[
                    iter]  # type cast from tf.float64 to tf.float16
                kl_bin.append(temp)
            kl_bin = torch.cat(kl_bin, dim=0).to(torch.float64)  # 256 x 4096

            attns_reshaped = attns.view(-1, 4096)
            kl_bin = kl_bin.to(torch.float64)
            attns_reshaped = attns_reshaped.to(torch.float64)
            # Perform matrix multiplication
            matmul_result = torch.matmul(kl_bin, attns_reshaped)
            # Compute the sum of kl_bin along dimension 1 and keep the dimensions
            kl_bin_sum = torch.sum(kl_bin, dim=1, keepdim=True)
            # Divide the result of the matrix multiplication by the sum
            result = matmul_result / kl_bin_sum
            # Reshape the result to (-1, 64, 64)
            new_attns = result.view(-1, 64, 64)  # 256 x 64 x 64

        else:
            # The rest of merging iterations, reducing the number of masks
            matched = set()
            new_attns = []
            for i, point in enumerate(attns):
                if i in matched:
                    continue
                matched.add(i)
                anchor = point
                kl_bin = (self.KL(anchor, attns) < kl_threshold[iter]).numpy()  # 64 x 64
                if kl_bin.sum() > 0:
                    matched_idx = np.arange(len(attns))[kl_bin.reshape(-1)]
                    for idx in matched_idx: matched.add(idx)
                    aggregated_attn = attns[kl_bin].mean(0)
                    new_attns.append(aggregated_attn.reshape(1, 64, 64))
            new_attns = torch.cat(new_attns, dim=0)
        return np.array(new_attns)

    def generate_masks(self, attns):
        for i in range(len(self.kl_threshold)):
            if i == 0:
                attns_merged = self.mask_merge(i, attns, self.kl_threshold, grid=self.grid)
            else:
                attns_merged = self.mask_merge(i, attns_merged, self.kl_threshold)

        # Kmeans refinement (optional for better visual consistency)
        if self.refine:
            attns = attns.reshape(-1, 64 * 64)
            kmeans = KMeans(n_clusters=attns_merged.shape[0], init=attns_merged.reshape(-1, 64 * 64), n_init=1).fit(
                attns)
            clusters = kmeans.labels_
            attns_merged = []
            for i in range(len(set(clusters))):
                cluster = (i == clusters)
                attns_merged.append(attns[cluster, :].mean(0).reshape(64, 64))
            attns_merged = np.array(attns_merged)

        # Upsampling
        attns_expanded = torch.from_numpy(attns_merged).unsqueeze(0)

        self.upsampled = F.interpolate(attns_expanded, scale_factor=8, mode='bilinear', align_corners=False).squeeze(0)

        # Non-Maximum Suppression
        argmax_index = torch.argmax(self.upsampled, dim=0)
        # Reshape the result to (512, 512)
        M_final = argmax_index.reshape(512, 512).cpu().numpy()
        return M_final

    def KL(self, x, Y):
        x = x.to(torch.float32)
        Y = Y.to(torch.float32)
        quotient = torch.log(x) - torch.log(Y)
        kl_1 = torch.sum(x * quotient, dim=(-2, -1)) / 2
        kl_2 = -torch.sum(Y * quotient, dim=(-2, -1)) / 2
        return kl_1 + kl_2

    def aggregate_weights(self, i, weight_ratio=None):
        if weight_ratio is None:
            weight_ratio = self.get_weight_ratio()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        aggre_weights = torch.zeros((64, 64, 64, 64), device=device)

        for index, (resolution, weights) in enumerate(self._self_attn.items()):
            weights = weights[i]
            size = int(np.sqrt(weights.shape[-1]))
            ratio = int(64 / size)
            # Average over the multi-head channel
            weights = weights.mean(0).reshape(-1, size, size)

            weights = weights.unsqueeze(0)
            weights = F.interpolate(weights, scale_factor=ratio, mode='bilinear', align_corners=True).squeeze(0)
            weights = weights.reshape(size, size, 64, 64)

            # Normalize to make sure each map sums to one
            weights = weights / torch.sum(weights, dim=(2, 3), keepdim=True)

            # Spatial tiling along the first two dimensions
            weights = torch.repeat_interleave(weights, repeats=ratio, dim=0)
            weights = torch.repeat_interleave(weights, repeats=ratio, dim=1)

            # Aggrgate accroding to weight_ratio
            aggre_weights += weights * weight_ratio[index]
        return aggre_weights.to(torch.float64).cpu().numpy()
        # return aggre_weights.numpy().astype(np.double)

    def get_weight_ratio(self, ):
        # This function assigns proportional aggergation weight
        sizes = []
        for resolution, weights in self._self_attn.items():
            sizes.append(np.sqrt(weights[0].shape[-2]))
        denom = np.sum(sizes)
        return sizes / denom

    def increment_T(self):
        self._T += 1

        self._layers_self = 0
        self._layers_cross = 0

    def increment_temp_T(self):
        self._temp_T += 1
