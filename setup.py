import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bafpipe",
    version="1.0.1",
    author="Lawrence Collins",
    author_email="l.j.@leeds.ac.uk",
    description="Automated deconvolution and analysis of Bruker mass spectra datasets using UniDec",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lawrencecollins/bafpipe",
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
    entry_points={
    'console_scripts': [
        '**bafpipe** = pipe:main',  # The command is 'bafpipe'
    ],}

)
