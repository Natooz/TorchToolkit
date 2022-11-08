from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='torchtoolkit',
    author='Nathan Fradet',
    url='https://github.com/Natooz/TorchToolkit',
    packages=find_packages(exclude=("test",)),
    version='0.0.2',
    license='MIT',
    description='Useful functions to use with PyTorch',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformer',
        'nlp'
    ],
    install_requires=[
        'torch>=1.10.0',
        'numpy>=1.19.0',
        'tqdm'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent"
    ]
)
