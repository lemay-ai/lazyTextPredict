import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lazy-text-predict",
    version="0.0.11",
    author="Edward Booker",
    author_email="epb378@gmail.com",
    description="Text classification automl",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lemay-ai/lazyTextPredict",
    packages=setuptools.find_packages(),
    install_requires =[
    "transformers",
    "nlp",
    "torch",
    "numpy",
    "scikit_learn",
    "sentencepiece",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.6',)
