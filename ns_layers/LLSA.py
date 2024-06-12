import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from claspy.segmentation import BinaryClaSPSegmentation


class LLSA(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, window_size=5, hidden_size=10,
                 change_point_threshold=0.1):
        super(LLSA, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.change_point_threshold = change_point_threshold

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            self.change_points = self._detect_change_points(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        assert not torch.isnan(self.mean).any(), "NaN found in mean calculation"
        assert not torch.isinf(self.mean).any(), "Inf found in mean calculation"
        assert not torch.isnan(self.stdev).any(), "NaN found in stdev calculation"
        assert not torch.isinf(self.stdev).any(), "Inf found in stdev calculation"

    def _normalize(self, x):
        change_points = self.change_points.int()

        if len(change_points.shape) == 1:
            change_points = change_points.unsqueeze(1).expand(-1, x.shape[1])

        x_normalized = x.clone()
        start_idx = 0

        segment_means = []
        segment_stds = []

        for i in range(1, change_points.shape[1]):  # Iterate over the sequence length
            if change_points[:, i].sum().item() > 0:  # Check if any batch has a change point at index i
                assert not torch.isnan(x).any(), "Input x contains NaNs"
                assert not torch.isinf(x).any(), "Input x contains infinite values"

                segment = x[:, start_idx:i, :]

                assert not torch.isnan(segment).any(), f"Segment contains NaNs at {start_idx}:{i}"
                assert not torch.isinf(segment).any(), f"Segment contains Infs at {start_idx}:{i}"

                dim2reduce = tuple(range(1, x.ndim - 1))

                segment_mean = segment.mean(1, keepdim=True).detach()
                segment_std = torch.sqrt(torch.var(segment, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

                if (segment_std == 0).any():
                    print(f"Zero std deviation found in segment {start_idx}:{i}, replacing with eps")
                    segment_std[segment_std == 0] = self.eps

                assert not torch.isnan(segment_mean).any(), f"Segment mean NaNs at {start_idx}:{i}"
                assert not torch.isnan(segment_std).any(), f"Segment std NaNs at {start_idx}:{i}"

                segment = (segment - segment_mean) / segment_std

                assert not torch.isnan(segment).any(), f"Segment NaNs at {start_idx}:{i}"

                if self.affine:
                    segment = segment * self.affine_weight + self.affine_bias
                x_normalized[:, start_idx:i, :] = segment

                segment_means.append(segment_mean)
                segment_stds.append(segment_std)

                start_idx = i

        # Normalize the remaining segment
        if start_idx < x.shape[1]:
            segment = x[:, start_idx:, :]

            assert not torch.isnan(segment).any(), f"Remaining segment contains NaNs at {start_idx}:end"
            assert not torch.isinf(segment).any(), f"Remaining segment contains Infs at {start_idx}:end"

            dim2reduce = tuple(range(1, x.ndim - 1))

            segment_mean = segment.mean(1, keepdim=True).detach()
            segment_std = torch.sqrt(torch.var(segment, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

            assert not torch.isnan(segment_mean).any(), f"Segment mean NaNs at {start_idx}:end"
            assert not torch.isnan(segment_std).any(), f"Segment std NaNs at {start_idx}:end"

            segment = (segment - segment_mean) / segment_std
            if self.affine:
                segment = segment * self.affine_weight + self.affine_bias
            x_normalized[:, start_idx:, :] = segment

            segment_means.append(segment_mean)
            segment_stds.append(segment_std)

        assert not torch.isnan(x_normalized).any(), "Normalized x contains NaNs"

        return x_normalized, segment_means, segment_stds

    def _denormalize(self, x):
        change_points = self.change_points.int()
        if len(change_points.shape) == 1:
            change_points = change_points.unsqueeze(1).expand(-1, x.shape[1])

        x_denormalized = x.clone()
        start_idx = 0

        for i in range(1, change_points.shape[1]):  # Iterate over the sequence length
            if change_points[:, i].sum().item() > 0:  # Check if any batch has a change point at index i
                segment = x[:, start_idx:i, :]

                # Calculate mean and standard deviation for the segment
                dim2reduce = tuple(range(1, x.ndim - 1))
                segment_mean = segment.mean(1, keepdim=True).detach()
                segment_std = torch.sqrt(torch.var(segment, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

                assert not torch.isnan(
                    segment_mean).any(), f"Segment mean NaNs at {start_idx}:{i}, shape_mean: {self.mean.shape}, shape_x: {x.shape}"
                assert not torch.isnan(
                    segment_std).any(), f"Segment std NaNs at {start_idx}:{i}, shape_std: {self.stdev.shape}, shape_x: {x.shape}"

                if self.affine:
                    segment = (segment - self.affine_bias) / (self.affine_weight + self.eps)
                segment = segment * segment_std + segment_mean

                x_denormalized[:, start_idx:i, :] = segment
                start_idx = i

        # Denormalize the remaining segment
        if start_idx < x.shape[1]:
            segment = x[:, start_idx:, :]
            dim2reduce = tuple(range(1, x.ndim - 1))

            segment_mean = segment.mean(1, keepdim=True).detach()
            segment_std = torch.sqrt(torch.var(segment, dim=1, keepdim=True, unbiased=False) + self.eps).detach()

            assert not torch.isnan(
                segment_mean).any(), f"Segment mean NaNs at {start_idx}:{i}, shape_mean: {self.mean.shape}, shape_x: {x.shape}"
            assert not torch.isnan(
                segment_std).any(), f"Segment std NaNs at {start_idx}:{i}, shape_std: {self.stdev.shape}, shape_x: {x.shape}"

            if self.affine:
                segment = (segment - self.affine_bias) / (self.affine_weight + self.eps)
            segment = segment * segment_std + segment_mean
            x_denormalized[:, start_idx:, :] = segment

        assert not torch.isnan(x_denormalized).any(), f"Denormalized x nan, denormalized_x: {x_denormalized}"

        return x_denormalized

    def _detect_change_points(self, x):
        batch_size, seq_len, num_features = x.shape
        change_points = torch.zeros(batch_size, seq_len, 1, dtype=torch.int32).to(x.device)

        for i in range(batch_size):
            for j in range(num_features):
                data = x[i, :, j].cpu().numpy().astype(np.float64)  # Ensure the data is in float64

                # Apply BinaryClaSPSegmentation algorithm from claspy library
                clasp = BinaryClaSPSegmentation(window_size=(self.window_size // 5))  # Adjust the window size as needed
                detected_cps = clasp.fit_predict(data)

                for point in detected_cps:
                    change_points[i, point, 0] = 1

        return change_points