"""Setup configuration for the package."""
from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="anomalous-market-behavior",
    version="2.0.0",
    author="James Olaitan",
    description="Anomalous Market Behavior Recognition with Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jamesolaitan/Anomalous-Market-Behavior-Recognition-with-Machine-Learning",
    packages=find_packages(exclude=["tests", "notebooks"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.28",
        "duckdb>=0.9.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.17.0",
        "streamlit>=1.28.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "pre-commit>=3.4.0",
            "mypy>=1.5.0",
        ],
    },
)

