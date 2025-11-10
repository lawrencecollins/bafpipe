import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BafPipe",

    version="0.3.4",

    author="Lawrence Collins",
    author_email="chmlco@leeds.ac.uk",
    description="Automated deconvolution of Bruker mass spectra datasets using UniDec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lawrencecollins/BafPipe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    setup_requires=['wheel'],
    install_requires=['unidec', 'seaborn'],

    package_data={'':['*.dll']},

)
