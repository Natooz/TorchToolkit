from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='flexformer',
    author='Nathan Fradet',
    url='https://github.com/Natooz/flexformer',
    packages=find_packages(exclude=("test",)),
    version='1.0.4',
    license='MIT',
    description='A general implementation of Transformer, to play around with attention and build custom architectures',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        'artificial intelligence',
        'deep learning',
        'transformer',
        'nlp'
    ],
    install_requires=[
        'torch>=1.11.0',
        'tqdm'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: MIDI',
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent"
    ]
)
