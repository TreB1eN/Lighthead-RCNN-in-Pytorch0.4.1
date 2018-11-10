from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='psroialign_cuda',
    ext_modules=[
        CUDAExtension('psroialign_cuda', [
            'psroialign_cuda.cpp',
            'psroialign_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
