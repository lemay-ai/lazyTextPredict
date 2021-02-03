# Lazy Text Predict

## Usage

You can currently upload data which has single categories (i.e. the models can be trained to detect differences between happy, jealous or sad text etc., but not both happy and excited). Your data should be submitted as python lists or pandas series to the fields Xdata and Ydata. Alternately you can pass csv or xlsx files to the appropriate options. 

Click here for an extensive example notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lemay-ai/lazyTextPredict/blob/master/examples/lazy_text_predict_example.ipynb)
, or see below:

```
from lazy-text-predict import basic_classification

trial=basic_classification.LTP(Xdata=X,Ydata=Y, csv=None, xlsx=None, x_col='X', y_col='Y', models='all') 
# Xdata is a list of text entries, and Ydata is a list of corresponding labels.
# csv and xlsx give options to load data from those file formats (you can pass the file or the file's location)
# x_col and y_col are strings that specify the columns of the # text and label columns in your csv or xlsx file respectively.
# You can choose between 'transformers'-based, 'count-vectorizer'-based, and 'all' models.

trial.run(training_epochs=5) 
#This trains the models specified above on the data you loaded. 
#Here you can specify the number of training epochs. 
#Fewer training epochs will give poorer performance, but will run quicker to allow debugging.

trial.print_metrics_table()
# This will return the performance of the models that have been trained:
                    Model            loss        accuracy              f1       precision          recall
        bert-base-uncased         0.80771         0.69004         0.68058          0.6783         0.69004
           albert-base-v2          0.8885         0.62252          0.6372           0.714         0.62252
             roberta-base         0.99342           0.533         0.56416         0.68716           0.533
               linear_SVM         0.36482         0.63518         0.30077         0.47439         0.30927
multinomial_naive_bayesian         0.31697         0.68303         0.35983           0.443         0.37341


trial.predict(text) 
# Here text is some custom, user-specified string that your trained classifiers can classify. 
# This will return the class's index based on how the order it appears in your input labels.
```
This will train and test each of the models show you their performance (loss rate, f1 score, training time, computing resources required etc.) and let you classify your own text.


The models are currently hard-coded, i.e. you can only choose between transformer and count-vectorizer models, but watch this space!

Once you have determined which model is best for your application you can do a more in-depth training on the model of your choice. This can be done by calling a new instance of the LTP class and running a focused training:

```
focused_trial=basic_classification.LTP(test_frac=0.05,train_frac=0.45)
focused_trial.run(focused=True,focused_model='bert-base-uncased',training_epochs=5)
```

We have added several example ipynb files to show how the library may be used.

## Installation

Install the package from PyPi in command line:
```
pip install lazy-text-predict
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

A good test is to try to install the package, if you can, there is a chance you could run it!

A practical alternative is to run this all in google colab pro or similar platforms that give you access to the resources you need (although these might not be free!).

