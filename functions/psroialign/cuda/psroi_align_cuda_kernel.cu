#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

__global__ void my_cuda_forward_kernel(
    const float* __restrict__ bottom_data,
    const float* __restrict__ bottom_rois,
    size_t total_size,
    float spatial_scale, 
    int channels,
    int height,
    int width,
    int pooled_dim,
    int pooled_height, 
    int pooled_width,
    int group_size,
    int sampling_ratio,
    float* __restrict__ top_data,
    int* __restrict__ argmax_data) {
    CUDA_KERNEL_LOOP(index, total_size)
    {
      int ph = (index / pooled_width) % pooled_height;
      int pw = index % pooled_width;
      int ctop = (index / pooled_width / pooled_height) % pooled_dim;
      int n = index / pooled_width / pooled_height / pooled_dim;

      float roi_start_w = static_cast<float>(round(bottom_rois[n * 4 + 0])) * spatial_scale;
      float roi_start_h = static_cast<float>(round(bottom_rois[n * 4 + 1])) * spatial_scale;
      float roi_end_w = static_cast<float>(round(bottom_rois[n * 4 + 2])) * spatial_scale;
      float roi_end_h = static_cast<float>(round(bottom_rois[n * 4 + 3])) * spatial_scale;

      // Force too small ROIs to be 1x1
      float roi_height = max(roi_end_h - roi_start_h, 0.1);
      float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

      // Compute w and h at bottom
      float bin_size_h = roi_height / static_cast<float>(pooled_height);
      float bin_size_w = roi_width / static_cast<float>(pooled_width);

      // Compute c at bottom
      int gh = floor(static_cast<float>(ph) * group_size / pooled_height);
      int gw = floor(static_cast<float>(pw) * group_size / pooled_width);
      gh = min(max(gh, 0), group_size - 1);
      gw = min(max(gw, 0), group_size - 1);
      int c = (ctop * group_size + gh) * group_size + gw;

      int bottom_data_offset = c * height * width;

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)? sampling_ratio : ceil(roi_height / pooled_height);  // e.g. = 2
      int roi_bin_grid_w = (sampling_ratio > 0)? sampling_ratio : ceil(roi_width / pooled_width);

      float maxval = -1E+20;
      int maxidx = -1;

      for (int iy = 0; iy < roi_bin_grid_h; iy++)
      // e.g. iy = 0, 1
      {
        float y = roi_start_h + ph * bin_size_h + static_cast<float>(iy + .5f) * bin_size_h / static_cast<float>(roi_bin_grid_h);  // e.g. 0.5, 1.5
        for (int ix = 0; ix < roi_bin_grid_w; ix++) {
          float x = roi_start_w + pw * bin_size_w + static_cast<float>(ix + .5f) * bin_size_w / static_cast<float>(roi_bin_grid_w);
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
            float v1 = bottom_data[bottom_data_offset + y_low * width + x_low];
            float v2 = bottom_data[bottom_data_offset + y_low * width + x_high];
            float v3 = bottom_data[bottom_data_offset + y_high * width + x_low];
            float v4 = bottom_data[bottom_data_offset + y_high * width + x_high];
            float w1 = hy * hx;
            float w2 = hy * lx;
            float w3 = ly * hx;
            float w4 = ly * lx;

            // }}

            float tmpval = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
            int bottom_index = iy * roi_bin_grid_w + ix;
            if (tmpval > maxval) {
              maxval = tmpval;
              maxidx = bottom_index;
            }
          }
      }
      top_data[index] = maxval;
      argmax_data[index] = maxidx;
    }
  }

__global__ void my_cuda_backward_kernel(
    const float* __restrict__ top_diff,
    const int* __restrict__ argmax_data,
    const float* __restrict__ bottom_rois,
    size_t total_size,
    float spatial_scale, 
    int channels,
    int height,
    int width,
    int pooled_dim,
    int pooled_height, 
    int pooled_width,
    int group_size,
    int sampling_ratio,
    float* __restrict__ bottom_diff) {
    CUDA_KERNEL_LOOP(index, total_size)
    {
      int ph = (index / pooled_width) % pooled_height;
      int pw = index % pooled_width;
      int ctop = (index / pooled_width / pooled_height) % pooled_dim;
      int n = index / pooled_width / pooled_height / pooled_dim;

      // Do not using rounding; this implementation detail is critical
      float roi_start_w = static_cast<float>(round(bottom_rois[n * 4 + 0])) * spatial_scale;
      float roi_start_h = static_cast<float>(round(bottom_rois[n * 4 + 1])) * spatial_scale;
      float roi_end_w = static_cast<float>(round(bottom_rois[n * 4 + 2])) * spatial_scale;
      float roi_end_h = static_cast<float>(round(bottom_rois[n * 4 + 3])) * spatial_scale;

      // Force too small ROIs to be 1x1
      float roi_height = max(roi_end_h - roi_start_h, 0.1);
      float roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0

      // Compute w and h at bottom
      float bin_size_h = roi_height / static_cast<float>(pooled_height);
      float bin_size_w = roi_width / static_cast<float>(pooled_width);

      // Compute c at bottom
      int gh = floor(static_cast<float>(ph) * group_size / pooled_height);
      int gw = floor(static_cast<float>(pw) * group_size / pooled_width);
      gh = min(max(gh, 0), group_size - 1);
      gw = min(max(gw, 0), group_size - 1);
      int c = (ctop * group_size + gh) * group_size + gw;

      int bottom_diff_offset = c * height * width;

      int top_offset = (n * pooled_dim + ctop) * pooled_height * pooled_width;
      float top_diff_this_bin = top_diff[top_offset + ph * pooled_width + pw];

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0)? sampling_ratio : ceil(roi_height / pooled_height); // e.g. = 2
      int roi_bin_grid_w = (sampling_ratio > 0)? sampling_ratio : ceil(roi_width / pooled_width);

      int maxidx = argmax_data[top_offset + ph * pooled_width + pw];
      int iy = maxidx / roi_bin_grid_w;
      int ix = maxidx % roi_bin_grid_w;

      float y = roi_start_h + ph * bin_size_h + static_cast<float>(iy + .5f) * bin_size_h / static_cast<float>(roi_bin_grid_h);  // e.g. 0.5, 1.5
      float x = roi_start_w + pw * bin_size_w + static_cast<float>(ix + .5f) * bin_size_w / static_cast<float>(roi_bin_grid_w);  

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

      float g1 = top_diff_this_bin * w1;
      float g2 = top_diff_this_bin * w2;
      float g3 = top_diff_this_bin * w3;
      float g4 = top_diff_this_bin * w4;

      if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
        atomicAdd(&bottom_diff[bottom_diff_offset + y_low * width + x_low], g1);
        atomicAdd(&bottom_diff[bottom_diff_offset + y_low * width + x_high], g2);
        atomicAdd(&bottom_diff[bottom_diff_offset + y_high * width + x_low], g3);
        atomicAdd(&bottom_diff[bottom_diff_offset + y_high * width + x_high], g4);
      }
        // }}
  } 
} // namespace

int my_cuda_forward(
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor top_data,
    at::Tensor argmax_data,
    float spatial_scale,
    int group_size,
    int sampling_ratio) {
    const auto channels = bottom_data.size(1);
    const auto height = bottom_data.size(2);
    const auto width = bottom_data.size(3);
    const auto batch_size = top_data.size(0);
    const auto pooled_dim = top_data.size(1);
    const auto pooled_height = top_data.size(2);
    const auto pooled_width = top_data.size(3);

    const auto total_size = batch_size * pooled_dim * pooled_height * pooled_width;

    const int threads = 1024;
    const int blocks = (total_size + threads - 1) / threads;

    my_cuda_forward_kernel<<<blocks, threads>>>(
      bottom_data.data<float>(),
      bottom_rois.data<float>(),
      total_size,
      spatial_scale, 
      channels,
      height,
      width,
      pooled_dim,
      pooled_height, 
      pooled_width,
      group_size,
      sampling_ratio,
      top_data.data<float>(),
      argmax_data.data<int>());

    return 1;
}

int my_cuda_backward(
    at::Tensor top_diff,
    at::Tensor argmax_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_diff,
    float spatial_scale,
    int group_size,
    int sampling_ratio) {    

    const auto channels = bottom_diff.size(1);
    const auto height = bottom_diff.size(2);
    const auto width = bottom_diff.size(3);
    const auto batch_size = top_diff.size(0);
    const auto pooled_dim = top_diff.size(1);
    const auto pooled_height = top_diff.size(2);
    const auto pooled_width = top_diff.size(3);
    const auto total_size = batch_size * pooled_dim * pooled_height * pooled_width;
  
    const int threads = 1024;
    const int blocks = (total_size + threads - 1) / threads;

    my_cuda_backward_kernel<<<blocks, threads>>>(
      top_diff.data<float>(),
      argmax_data.data<int>(),
      bottom_rois.data<float>(),
      total_size,
      spatial_scale, 
      channels,
      height,
      width,
      pooled_dim,
      pooled_height, 
      pooled_width,
      group_size,
      sampling_ratio,
      bottom_diff.data<float>());

  return 1;
}
