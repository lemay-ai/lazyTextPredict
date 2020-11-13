# lazy Text Predict

Do you want to automatically tag your blog posts? Identify scientific terms in a document? Try to identify the author of a new novel? These are all text classification problems, but may require different levels of complexity in their execution. You don't want to use a deep neural network when a decision tree could suffice, and you also don't want to get stuck trying to get a

This tool lets you quickly choose between different natural language processing tools for your classification problem.

What we do is load your text for classification into several of the most useful tools we have identified, train the tools on a small portion of the data, and then try to show you which would be best for your problem.

The models we use at the moment come from [this](https://github.com/huggingface/transformers) library.

The results show you metrics like Accuracy, f1 score, precision, recall etc. in a simple table-like output.
You can then go ahead and choose whiever model works best for you.

This tool is built on top of [PyTorch](https://pytorch.org/) framework and [transformers](https://github.com/huggingface/transformers) library.

## System Requirements

Unfortunately, this tool requires a fair bit of computing power. If you do not have a GPU that the tool can use, you will struggle to run it.

A good test is to try to install tensorflow-gpu, if you can, there is a chance you could run this tool!

A practical alternative is to run this all in google colab pro or similar platforms that give you access to the resources you need.

## Installation

Install the package from the PyPi test server:
```
python3 -m pip install --index-url https://test.pypi.org/simple/ lazytextpredict-test
```

## Usage (temporary)

Currently the data and models are hard-coded.
```
from lazytextpredict import basic_classification

basic_classification.main()
```
