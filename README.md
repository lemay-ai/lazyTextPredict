# lazy Text Predict

Do you want to automatically tag your blog posts? Identify scientific terms in a document? Try to identify the author of a new novel? These are all text classification problems, but may require different levels of complexity in their execution. You don't want to use a deep neural network when a decision tree could suffice, or vice-versa!

This tool lets you quickly choose between different natural language processing tools for your classification problem.

What we do is load your text for classification into several of the most useful tools we have identified, train the tools on a small portion of the data, and then try to show you which would be best for your problem.

The models we use at the moment come from [this](https://github.com/huggingface/transformers) library.

The results show you metrics like Accuracy, f1 score, precision, recall etc. in a simple table-like output.
You can then go ahead and choose whiever model works best for you.

This tool is built on top of [PyTorch](https://pytorch.org/) framework and [transformers](https://github.com/huggingface/transformers) library.

## System Requirements

Unfortunately, this tool requires a fair bit of computing power. If you do not have a GPU that the tool can use, you will struggle to run it.

A good test is to try to install tensorflow-gpu, if you can, there is a chance you could run this tool!
```
pip install tensorflow-gpu
```

A practical alternative is to run this all in google colab pro or similar platforms that give you access to the resources you need (although these might not be free!).

## Installation

Install the package from the PyPi test server in command line:
```
python3 -m pip install --index-url https://test.pypi.org/simple/ lazytextpredict-test
```

## Usage

Currently the data and models are hard-coded, i.e. you can't upload your own data yet or choose your models, but watch this space!

```
from lazytextpredict import basic_classification

basic_classification.main()
```
This will train and test each of the models show you their performance (loss rate, f1 score, training time, computing resources required etc.)
