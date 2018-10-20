#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

int PSROIAlignForwardLaucher(
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor top_data,
    at::Tensor argmax_data,
    float spatial_scale,
    int group_size,
    int sampling_ratio);

int PSROIAlignBackwardLaucher(
    at::Tensor top_diff,
    at::Tensor argmax_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_diff,
    float spatial_scale,
    int group_size,
    int sampling_ratio);

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int psroi_align_forward_cuda(
    at::Tensor bottom_data,
    at::Tensor bottom_rois,
    at::Tensor top_data,
    at::Tensor argmax_data,
    float spatial_scale,
    int group_size,
    int sampling_ratio) {

    CHECK_INPUT(bottom_data);
    CHECK_INPUT(bottom_rois);
    CHECK_INPUT(top_data);
    CHECK_INPUT(argmax_data);

    int size_rois = bottom_rois.size(1);

    if (size_rois != 4) {
        printf("wrong roi size\n");
        return 0;
    }

    PSROIAlignForwardLaucher(bottom_data, bottom_rois, top_data, argmax_data, spatial_scale, group_size, sampling_ratio); 
    return 1;
}

int psroi_align_backward_cuda(
    at::Tensor top_diff,
    at::Tensor argmax_data,
    at::Tensor bottom_rois,
    at::Tensor bottom_diff,
    float spatial_scale,
    int group_size,
    int sampling_ratio) {

    CHECK_INPUT(top_diff);
    CHECK_INPUT(bottom_rois);
    CHECK_INPUT(bottom_diff);
    CHECK_INPUT(argmax_data);

    int size_rois = bottom_rois.size(1);

    if (size_rois != 4) {
        printf("wrong roi size\n");
        return 0;
    }

    PSROIAlignBackwardLaucher(top_diff, argmax_data, bottom_rois, bottom_diff, spatial_scale, group_size, sampling_ratio);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &psroi_align_forward_cuda, "PSRoi_Align forward (CUDA)");
  m.def("backward", &psroi_align_backward_cuda, "PSRoi_Align backward (CUDA)");
}
