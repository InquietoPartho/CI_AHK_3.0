# setup.py
import setuptools

setuptools.setup(
    name="ci_ahk",
    version="0.1.0",
    author="Pijush Kanti Roy Partho",
    author_email="pijushkantiroy2040@gmail.com",
    description="Class-Imbalance–Aware Adaptive Hybrid Kernel for SVM",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/InquietoPartho/CI_AHK_3.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20",
        "scikit-learn>=1.1",
        "scipy>=1.7",
        "statsmodels>=0.13"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
