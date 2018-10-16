#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

int my_cuda_forward(
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor top_data,
    at::Tensor argmax_data,
    float spatial_scale,
    int group_size,
    int sampling_ratio);

int my_cuda_backward(
    at::Tensor top_diff,
    at::Tensor argmax_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_diff,
    float spatial_scale,
    int group_size,
    int sampling_ratio);

int my_forward(
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor top_data,
    at::Tensor argmax_data,
    float spatial_scale,
    int group_size,
    int sampling_ratio) {

  return my_cuda_forward(bottom_data, bottom_rois, top_data, argmax_data, spatial_scale, group_size, sampling_ratio); 
}

int my_backward(
    at::Tensor top_diff,
    at::Tensor argmax_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_diff,
    float spatial_scale,
    int group_size,
    int sampling_ratio) {

  return my_cuda_backward(top_diff, argmax_data, bottom_rois, bottom_diff, spatial_scale, group_size, sampling_ratio);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &my_forward, "my forward (CUDA)");
  m.def("backward", &my_backward, "my backward (CUDA)");
}
