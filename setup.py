import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hc-nlp",
    version="0.3.4",
    author="Science Museum Group",
    description="Heritage Connector NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheScienceMuseum/heritage-connector-nlp",
    download_url="https://github.com/TheScienceMuseum/heritage-connector-nlp/archive/v0.3.0.tar.gz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["spacy-nightly==3.0.0rc3", "spacy-transformers==1.0.0rc2"],
    packages=["hc_nlp"],
)
