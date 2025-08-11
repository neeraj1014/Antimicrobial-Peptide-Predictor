from setuptools import setup, find_packages

setup(
    name="amppred",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub[hf_xet]",
        "torch",
        "transformers", 
        "pandas",
        "numpy",
        "pyrodigal",
        "biopython"
    ],
    entry_points={
        'console_scripts': [
            'amp-run=amppred.run:main',
            'amp-build=amppred.build:main',
        ],
    },
    python_requires=">=3.8",
    author="Neeraj Kumar Singh",
    author_email="neeraj.nks1001@gmail.com",
    description="Antimicrobial peptide prediction tool",
    long_description_content_type="text/markdown",
)