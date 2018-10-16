from torch.utils.cpp_extension import load
my_cuda = load(
    'psroi_align_cuda', ['psroi_align_cuda.cpp', 'psroi_align_cuda_kernel.cu'], verbose=True)
help(my_cuda)
