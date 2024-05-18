from setuptools import setup, Extension
from torch.utils import cpp_extension

torch_library_paths = cpp_extension.library_paths(cuda=False)

ext_modules = [
    cpp_extension.CppExtension(
        name='torch_persistent_homology.persistent_homology_cpu',
        sources=['torch_persistent_homology/persistent_homology_cpu.cpp'],
        extra_link_args=[
            '-Wl,-rpath,' + library_path for library_path in torch_library_paths
        ]
    )
]

setup(
    name='torch_persistent_homology',
    version='0.1',
    packages=['torch_persistent_homology'],
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)
