# lazy Text Predict

A simple tool built to train and test multiple transformer-models in one go.

The aim here is to let you input any dataset on which you want to do classification, and this tool automatically trains the available transformer models in [this](https://github.com/huggingface/transformers) library, tests each of them and gives you the metrics like Accuracy, f1 score, precision, recall etc. in a simple table-like output.
You can then go ahead and choose whiever model works best for you.

This tool is built on top of [PyTorch](https://pytorch.org/) framework and [transformers](https://github.com/huggingface/transformers) library.

## Installation (temporary)

Install dependencies manually:
```
pip install transformers
```
```
pip install nlp
```

Install the package from the PyPi test server:
```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps lazytextpredict-test
```

## Usage (temporary)

Currently the data and models are hard-coded.
```
from lazytextpredict import basic_classification

basic_classification.main()
```