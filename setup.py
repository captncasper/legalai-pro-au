import numpy as np
from setuptools import setup, find_packages

setup(
    name="aussie-legal-ai",
    version="1.0.0",
    author="Your Name",
    description="Australian Legal AI - Semantic Search System",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "faiss-cpu>=1.7.4",
        "fastapi>=0.100.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.8",
)
