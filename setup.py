import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hc-nlp",
    version="0.2.1",
    author="Science Museum Group",
    description="Heritage Connector NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheScienceMuseum/heritage-connector-nlp",
    download_url="https://github.com/TheScienceMuseum/heritage-connector-nlp/archive/v0.1.0.tar.gz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["spacy>=2.3.0,<2.4.0", "tqdm>=4.10.0,<5.0.0"],
    packages=["hc_nlp"],
)
