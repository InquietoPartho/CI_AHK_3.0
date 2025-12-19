from setuptools import setup, find_packages

setup(
    name="adaptive-hybrid-kernel",
    version="0.3.0",
    description="Adaptive Hybrid Kernel (AHK) for Support Vector Machines",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Pijush Kanti Roy Partho",
    author_email="pijushkantiroy2040@gmail.com",
    url="https://github.com/InquietoPartho/CI_AHK_3.0",  # optional
    license="MIT",  # or Apache-2.0, GPL-3.0 depending on your choice
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "scikit-learn>=1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
