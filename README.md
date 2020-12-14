# Lazy Text Predict

## Usage

Currently the models are hard-coded, i.e. you can only choose between neural network and count-vectorizer models, but watch this space!

You can currently only upload data which has single categories (i.e. the models can be trained to detect differences between happy, jealous or sad text etc., but not both happy and excited). Your data should be submitted as python lists to the fields Xdata and Ydata.

```
from lazytextpredict import basic_classification

trial=basic_classification.LTP() 

trial.run(Xdata, Ydata, models='all') 
# Xdata is a list of text entries, and Ydata is a list of corresponding labels.
# You can choose between 'cnn'-based, 'count-vectorizer'-based, and 'all' models.

trial.print_metrics_table()
# This will return the performance of the models that have been trained.


trial.predict(text) 
# Here text is some custom, user specified string that your trained classifiers can classify. 
# This will return the class's index based on how the order it appears in your input labels.
```
This will train and test each of the models show you their performance (loss rate, f1 score, training time, computing resources required etc.)

## Installation

Install the package from the PyPi test server in command line:
```
python3 -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple  lazytextpredict-installs
```

## About

Do you want to automatically tag your blog posts? Identify scientific terms in a document? Try to identify the author of a new novel? These are all text classification problems, but may require different levels of complexity in their execution. You don't want to use a deep neural network when a decision tree could suffice, or vice-versa!

How do you choose the best option out of so many choices?

![How to choose out of seemingly identical choices?](https://cdn.pixabay.com/photo/2016/08/15/08/40/apple-1594742_960_720.jpg)

This tool lets you quickly choose between different natural language processing tools for your classification problem.

What we do is load your text for classification into several of the most useful tools we have identified, train the tools on a small portion of the data, and then try to show you which would be best for your problem.

The models we use at the moment come from [this](https://github.com/huggingface/transformers) library.

The results show you metrics like: accuracy, f1 score, precision, recall etc. in a simple table-like output.
You can then go ahead and choose whichever model works best for you.

This tool is built on top of [PyTorch](https://pytorch.org/) framework and [transformers](https://github.com/huggingface/transformers) library. The inspiration for this tool came from the great [Lazy Predict](https://pypi.org/project/lazypredict/) package, which you should check out if you are interested.

## System Requirements

Unfortunately, this tool requires a fair bit of computing power. If you do not have a GPU that the tool can use, you will struggle to run it.

A good test is to try to install tensorflow-gpu, if you can, there is a chance you could run this tool!
```
pip install tensorflow-gpu
```

A practical alternative is to run this all in google colab pro or similar platforms that give you access to the resources you need (although these might not be free!).


