from __future__ import division

import numpy as np
import six

from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check
import pdb

if cuda.available:
    import cupy as cp


class PSROIMaxAlign2D(function.Function):

    def __init__(
            self, out_c, out_h, out_w, spatial_scale,
            group_size, sampling_ratio=-1):
        self.out_c, self.out_h, self.out_w = out_c, out_h, out_w
        print('out_c : {}, out_h : {}, out_w : {}'.format(out_c, out_h, out_w))
        self.spatial_scale = spatial_scale
        print('spatial_scale : {}'.format(spatial_scale))
        self.group_size = group_size
        print('group_size : {}'.format(group_size))
        self.sampling_ratio = sampling_ratio
        print('sampling_ratio : {}'.format(sampling_ratio))

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x_type, roi_type, roi_index_type = in_types
        type_check.expect(
            x_type.dtype == np.float32,
            x_type.ndim == 4,
            roi_type.dtype == np.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 4,
            roi_index_type.dtype == np.int32,
            roi_index_type.ndim == 1,
            roi_type.shape[0] == roi_index_type.shape[0]
        )

    def forward_cpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
#         print('bottom_data.shape : {}'.format(bottom_data.shape))
#         print('bottom_rois.shape : {}'.format(bottom_rois.shape))
#         print('bottom_roi_indices.shape : {}'.format(bottom_roi_indices.shape))
        channels, height, width = bottom_data.shape[1:]
#         print('channels :{}, height : {}, width : {}'.format(channels, height, width))
        n_roi = bottom_rois.shape[0]
#         print('n_roi : {}'.format(sampling_ratio))
        top_data = np.empty(
            (n_roi, self.out_c, self.out_h, self.out_w), dtype=np.float32)
        self.argmax_data = np.empty(top_data.shape, dtype=np.int32)

        group_size = self.group_size
        pooled_dim, pooled_width, pooled_height \
            = self.out_c, self.out_w, self.out_h
        spatial_scale = self.spatial_scale

        for i in six.moves.range(top_data.size):
            pw = i % pooled_width
            ph = int(i / pooled_width) % pooled_height
            ctop = int(i / pooled_width / pooled_height) % pooled_dim
            n = int(i / pooled_width / pooled_height / pooled_dim)

            roi_batch_ind = int(bottom_roi_indices[n])
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_height = max(roi_end_h - roi_start_h, 1.)
            roi_width = max(roi_end_w - roi_start_w, 1.)
            bin_size_h = 1. * roi_height / pooled_height
            bin_size_w = 1. * roi_width / pooled_width

            gh = np.floor(float(ph) * group_size / pooled_height)
            gw = np.floor(float(pw) * group_size / pooled_width)
            gh = int(min(max(gh, 0), group_size - 1))
            gw = int(min(max(gw, 0), group_size - 1))
            c = (ctop * group_size + gh) * group_size + gw

            if self.sampling_ratio > 0:
                roi_bin_grid_h = self.sampling_ratio
                roi_bin_grid_w = self.sampling_ratio
            else:
                roi_bin_grid_h = np.ceil(roi_height / pooled_height)
                roi_bin_grid_w = np.ceil(roi_width / pooled_width)

            maxval = -1e20
            maxidx = -1
            iy = 0
            while iy < roi_bin_grid_h:
                y = roi_start_h + ph * bin_size_h + \
                    (iy + .5) * bin_size_h / roi_bin_grid_h
                ix = 0
                while ix < roi_bin_grid_w:
                    x = roi_start_w + pw * bin_size_w + \
                        (ix + .5) * bin_size_w / roi_bin_grid_w

                    # bilinear interpolation {{
                    if y < -1 or y > height or x < -1 or x > width:
                        # empty
                        continue

                    if y <= 0:
                        y = 0
                    if x <= 0:
                        x = 0

                    y_low = int(y)
                    x_low = int(x)

                    if y_low >= height - 1:
                        y_high = y_low = height - 1
                        y = float(y_low)
                    else:
                        y_high = y_low + 1

                    if x_low >= width - 1:
                        x_high = x_low = width - 1
                        x = float(x_low)
                    else:
                        x_high = x_low + 1

                    ly = y - y_low
                    lx = x - x_low
                    hy = 1. - ly
                    hx = 1. - lx

                    v1 = bottom_data[roi_batch_ind, c, y_low, x_low]
                    v2 = bottom_data[roi_batch_ind, c, y_low, x_high]
                    v3 = bottom_data[roi_batch_ind, c, y_high, x_low]
                    v4 = bottom_data[roi_batch_ind, c, y_high, x_high]

                    w1 = hy * hx
                    w2 = hy * lx
                    w3 = ly * hx
                    w4 = ly * lx

                    tmpval = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                    bottom_index = iy * roi_bin_grid_w + ix
                    if (tmpval > maxval):
                        maxval = tmpval
                        maxidx = bottom_index

                    ix += 1
                iy += 1

            top_data[n, ctop, ph, pw] = maxval
            self.argmax_data[n, ctop, ph, pw] = maxidx

        return top_data,

    def forward_gpu(self, inputs):
        self.retain_inputs((1, 2))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = bottom_data.shape[1:]
        n_roi = bottom_rois.shape[0]
        top_data = cp.empty(
            (n_roi, self.out_c, self.out_h, self.out_w), dtype=np.float32)
        self.argmax_data = cuda.cupy.empty(top_data.shape, np.int32)
        cuda.elementwise(
            '''
            raw float32 bottom_data, raw float32 bottom_rois,
            raw int32 bottom_roi_indices,
            float32 spatial_scale, int32 channels,
            int32 height, int32 width,
            int32 pooled_dim, int32 pooled_height, int32 pooled_width,
            int32 group_size, int32 sampling_ratio
            ''',
            'float32 top_data, int32 argmax_data',
            '''
            // pos in output filter
            int ph = (i / pooled_width) % pooled_height;
            int pw = i % pooled_width;
            int ctop = (i / pooled_width / pooled_height) % pooled_dim;
            int n = i / pooled_width / pooled_height / pooled_dim;

            int roi_batch_ind = bottom_roi_indices[n];
            float roi_start_h = static_cast<float>(
                round(bottom_rois[n * 4 + 0])) * spatial_scale;
            float roi_start_w = static_cast<float>(
                round(bottom_rois[n * 4 + 1])) * spatial_scale;
            float roi_end_h = static_cast<float>(
                round(bottom_rois[n * 4 + 2])) * spatial_scale;
            float roi_end_w = static_cast<float>(
                round(bottom_rois[n * 4 + 3])) * spatial_scale;

            // Force too small ROIs to be 1x1
            float roi_height = max(roi_end_h - roi_start_h, 0.1);
            float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

            // Compute w and h at bottom
            float bin_size_h = roi_height / static_cast<float>(pooled_height);
            float bin_size_w = roi_width / static_cast<float>(pooled_width);

            // Compute c at bottom
            int gh = floor(
                static_cast<float>(ph) * group_size / pooled_height);
            int gw = floor(
                static_cast<float>(pw) * group_size / pooled_width);
            gh = min(max(gh, 0), group_size - 1);
            gw = min(max(gw, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int bottom_data_offset =
                (roi_batch_ind * channels + c) * height * width;

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_height / pooled_height);  // e.g. = 2
            int roi_bin_grid_w = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_width / pooled_width);

            float maxval = -1E+20;
            int maxidx = -1;
            for (int iy = 0; iy < roi_bin_grid_h; iy++)  // e.g. iy = 0, 1
            {
                float y = roi_start_h + ph * bin_size_h +
                    static_cast<float>(iy + .5f) * bin_size_h /
                        static_cast<float>(roi_bin_grid_h);  // e.g. 0.5, 1.5
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    float x = roi_start_w + pw * bin_size_w +
                        static_cast<float>(ix + .5f) * bin_size_w /
                            static_cast<float>(roi_bin_grid_w);

                    // bilinear_interpolation {{

                    // deal with cases that inverse elements are
                    // out of feature map boundary
                    if (y < -1. || y > height || x < -1. || x > width) {
                        // empty
                        continue;
                    }

                    if (y <= 0) {
                        y = 0;
                    }
                    if (x <= 0) {
                        x = 0;
                    }

                    int y_low = (int)y;
                    int x_low = (int)x;
                    int y_high;
                    int x_high;

                    if (y_low >= height - 1) {
                        y_high = y_low = height - 1;
                        y = (float)y_low;
                    } else {
                        y_high = y_low + 1;
                    }

                    if (x_low >= width - 1) {
                        x_high = x_low = width - 1;
                        x = (float)x_low;
                    } else {
                        x_high = x_low + 1;
                    }

                    float ly = y - y_low;
                    float lx = x - x_low;
                    float hy = 1. - ly;
                    float hx = 1. - lx;
                    // do bilinear interpolation
                    float v1 = bottom_data[bottom_data_offset +
                                           y_low * width + x_low];
                    float v2 = bottom_data[bottom_data_offset +
                                           y_low * width + x_high];
                    float v3 = bottom_data[bottom_data_offset +
                                           y_high * width + x_low];
                    float v4 = bottom_data[bottom_data_offset +
                                           y_high * width + x_high];
                    float w1 = hy * hx;
                    float w2 = hy * lx;
                    float w3 = ly * hx;
                    float w4 = ly * lx;

                    // }}

                    float tmpval = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
                    int bottom_index = iy * roi_bin_grid_w + ix;
                    if (tmpval > maxval) {
                        maxval = tmpval;
                        maxidx =  bottom_index;
                    }
                }
            }
            top_data = maxval;
            argmax_data = maxidx;
            ''', 'psroi_max_align_2d_fwd'
        )(bottom_data, bottom_rois, bottom_roi_indices,
          self.spatial_scale, channels, height, width,
          self.out_c, self.out_h, self.out_w,
          self.group_size, self.sampling_ratio,
          top_data, self.argmax_data)

        return top_data,

    def backward_cpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = np.zeros(self._bottom_data_shape, np.float32)

        spatial_scale = self.spatial_scale
        pooled_dim = self.out_c
        pooled_height = self.out_h
        pooled_width = self.out_w
        group_size = self.group_size
        top_diff = gy[0]

        for i in six.moves.range(top_diff.size):
            pw = i % pooled_width
            ph = int(i / pooled_width) % pooled_height
            ctop = int(i / pooled_width / pooled_height) % pooled_dim
            n = int(i / pooled_width / pooled_height / pooled_dim)

            roi_batch_ind = int(bottom_roi_indices[n])
            roi_start_h = bottom_rois[n, 0] * spatial_scale
            roi_start_w = bottom_rois[n, 1] * spatial_scale
            roi_end_h = bottom_rois[n, 2] * spatial_scale
            roi_end_w = bottom_rois[n, 3] * spatial_scale

            roi_width = max(roi_end_w - roi_start_w, 1.)
            roi_height = max(roi_end_h - roi_start_h, 1.)
            bin_size_h = 1. * roi_height / pooled_height
            bin_size_w = 1. * roi_width / pooled_width

            gh = np.floor(float(ph) * group_size / pooled_height)
            gw = np.floor(float(pw) * group_size / pooled_width)
            gh = int(min(max(gh, 0), group_size - 1))
            gw = int(min(max(gw, 0), group_size - 1))
            c = (ctop * group_size + gh) * group_size + gw

            top_diff_this_bin = top_diff[n, ctop, ph, pw]

            if self.sampling_ratio > 0:
                roi_bin_grid_h = self.sampling_ratio
                roi_bin_grid_w = self.sampling_ratio
            else:
                roi_bin_grid_h = np.ceil(roi_height / pooled_height)
                roi_bin_grid_w = np.ceil(roi_width / pooled_width)

            maxidx = self.argmax_data[n, ctop, ph, pw]
            iy = int(maxidx / roi_bin_grid_w)
            ix = maxidx % roi_bin_grid_w

            y = roi_start_h + ph * bin_size_h + \
                (iy + .5) * bin_size_h / roi_bin_grid_h
            x = roi_start_w + pw * bin_size_w + \
                (ix + .5) * bin_size_w / roi_bin_grid_w

            # bilinear_interpolation_gradient {{
            if y < -1 or y > height or x < -1 or x > width:
                # empty
                continue

            if y <= 0:
                y = 0
            if x <= 0:
                x = 0

            y_low = int(y)
            x_low = int(x)

            if y_low >= height - 1:
                y_high = y_low = height - 1
                y = float(y_low)
            else:
                y_high = y_low + 1

            if x_low >= width - 1:
                x_high = x_low = width - 1
                x = float(x_low)
            else:
                x_high = x_low + 1

            ly = y - y_low
            lx = x - x_low
            hy = 1. - ly
            hx = 1. - lx

            w1 = hy * hx
            w2 = hy * lx
            w3 = ly * hx
            w4 = ly * lx
            # }}

            g1 = top_diff_this_bin * w1
            g2 = top_diff_this_bin * w2
            g3 = top_diff_this_bin * w3
            g4 = top_diff_this_bin * w4

            if (x_low >= 0 and x_high >= 0 and
                    y_low >= 0 and y_high >= 0):
                bottom_diff[roi_batch_ind, c, y_low, x_low] += g1
                bottom_diff[roi_batch_ind, c, y_low, x_high] += g2
                bottom_diff[roi_batch_ind, c, y_high, x_low] += g3
                bottom_diff[roi_batch_ind, c, y_high, x_high] += g4

        return bottom_diff, None, None

    def backward_gpu(self, inputs, gy):
        _, bottom_rois, bottom_roi_indices = inputs
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, np.float32)
        cuda.elementwise(
            '''
            raw float32 top_diff, raw int32 argmax_data,
            raw float32 bottom_rois, raw int32 bottom_roi_indices,
            float32 spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_dim, int32 pooled_height, int32 pooled_width,
            int32 group_size, int32 sampling_ratio
            ''',
            'raw float32 bottom_diff',
            '''
            // (n, c, h, w) coords in bottom data
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int ctop = (i / pooled_width / pooled_height) % pooled_dim;
            int n = i / pooled_width / pooled_height / pooled_dim;

            // Do not using rounding; this implementation detail is critical
            int roi_batch_ind = bottom_roi_indices[n];
            float roi_start_h = static_cast<float>(
                round(bottom_rois[n * 4 + 0])) * spatial_scale;
            float roi_start_w = static_cast<float>(
                round(bottom_rois[n * 4 + 1])) * spatial_scale;
            float roi_end_h = static_cast<float>(
                round(bottom_rois[n * 4 + 2])) * spatial_scale;
            float roi_end_w = static_cast<float>(
                round(bottom_rois[n * 4 + 3])) * spatial_scale;

            // Force too small ROIs to be 1x1
            float roi_height = max(roi_end_h - roi_start_h, 0.1);
            float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

            // Compute w and h at bottom
            float bin_size_h = roi_height / static_cast<float>(pooled_height);
            float bin_size_w = roi_width / static_cast<float>(pooled_width);

            // Compute c at bottom
            int gh = floor(
                static_cast<float>(ph) * group_size / pooled_height);
            int gw = floor(
                static_cast<float>(pw) * group_size / pooled_width);
            gh = min(max(gh, 0), group_size - 1);
            gw = min(max(gw, 0), group_size - 1);
            int c = (ctop * group_size + gh) * group_size + gw;

            int bottom_diff_offset =
                (roi_batch_ind * channels + c) * height * width;

            int top_offset =
                (n * pooled_dim + ctop) * pooled_height * pooled_width;
            float top_diff_this_bin =
                top_diff[top_offset + ph * pooled_width + pw];

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_height / pooled_height); // e.g. = 2
            int roi_bin_grid_w = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_width / pooled_width);

            int maxidx = argmax_data[top_offset + ph * pooled_width + pw];
            int iy = maxidx / roi_bin_grid_w;
            int ix = maxidx % roi_bin_grid_w;

            float y = roi_start_h + ph * bin_size_h +
                static_cast<float>(iy + .5f) * bin_size_h /
                    static_cast<float>(roi_bin_grid_h);  // e.g. 0.5, 1.5
            float x = roi_start_w + pw * bin_size_w +
                static_cast<float>(ix + .5f) * bin_size_w /
                    static_cast<float>(roi_bin_grid_w);

            float w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;

            // bilinear_interpolation_gradient {{

            // deal with cases that inverse elements are
            // out of feature map boundary
            if (y < -1. || y > height || x < -1. || x > width) {
                // empty
                continue;
            }

            if (y <= 0) {
                y = 0;
            }
            if (x <= 0) {
                x = 0;
            }

            y_low = (int)y;
            x_low = (int)x;

            if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (float)y_low;
            } else {
                y_high = y_low + 1;
            }

            if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (float)x_low;
            } else {
                x_high = x_low + 1;
            }

            float ly = y - y_low;
            float lx = x - x_low;
            float hy = 1. - ly;
            float hx = 1. - lx;

            w1 = hy * hx;
            w2 = hy * lx;
            w3 = ly * hx;
            w4 = ly * lx;

            // }}

            float g1 = top_diff_this_bin * w1;
            float g2 = top_diff_this_bin * w2;
            float g3 = top_diff_this_bin * w3;
            float g4 = top_diff_this_bin * w4;

            if (x_low >= 0 && x_high >= 0 &&
                    y_low >= 0 && y_high >= 0) {
                atomicAdd(&bottom_diff[bottom_diff_offset +
                                       y_low * width + x_low], g1);
                atomicAdd(&bottom_diff[bottom_diff_offset +
                                       y_low * width + x_high], g2);
                atomicAdd(&bottom_diff[bottom_diff_offset +
                                       y_high * width + x_low], g3);
                atomicAdd(&bottom_diff[bottom_diff_offset +
                                       y_high * width + x_high], g4);
            }
            ''', 'psroi_max_align_2d_bwd'
        )(gy[0], self.argmax_data, bottom_rois, bottom_roi_indices,
          self.spatial_scale, channels, height, width,
          self.out_c, self.out_h, self.out_w,
          self.group_size, self.sampling_ratio, bottom_diff, size=gy[0].size)

        return bottom_diff, None, None


def psroi_max_align_2d(
        x, rois, roi_indices, out_c, out_h, out_w,
        spatial_scale, group_size, sampling_ratio=-1
):
    return PSROIMaxAlign2D(
        out_c, out_h, out_w, spatial_scale,
        group_size, sampling_ratio)(x, rois, roi_indices)
