from setuptools import setup, find_packages

setup(
    name="patent-revocation-prediction",
    version="0.1",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.0.0",
        "openpyxl>=3.0.0"
    ],
)