from setuptools import setup, find_packages

setup(
    name="wimfo",
    version="0.1.0",
    description="A package for computing W- and M-information.",
    author="Alberto Liardi",
    author_email='a.liardi@imperial.ac.uk',
    packages=find_packages(where="wimfo"),
    install_requires=["numpy", "matplotlib", "jupyter", "torch", "parametrization_cookbook"],
    zip_safe=False,
)
