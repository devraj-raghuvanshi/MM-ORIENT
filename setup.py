"""
Setup file for MM-ORIENT project
Multi-modal Sentiment Analysis using Graph Neural Networks
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mm-orient",
    version="1.0.0",
    author="MM-ORIENT Team",
    description="Multi-modal Sentiment Analysis using Graph Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.8.0",
        "tensorflow-hub>=0.12.0",
        "keras>=2.8.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "spektral>=1.2.0",
        "scikit-image>=0.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "jupyter>=1.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mm-orient-train=src.model:main",
        ],
    },
) 