import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lazytextpredict-test", # Replace with your own username
    version="0.0.3",
    author="Jitesh Pabla",
    author_email="jiteshpabla97@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lemay-ai/lazyTextPredict",
    packages=setuptools.find_packages(),
    install_requires =[
    "transformers==3.5.1",
    "nlp==0.4.0",
    "torch"==1.7.0+cu101",
    "numpy==1.18.5",
    "scikit_learn==0.23.2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
