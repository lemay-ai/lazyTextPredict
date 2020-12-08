import pandas as pd
import gc
import transformers
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from nlp import load_dataset, Dataset
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, hamming_loss
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from random import sample, choices
from joblib import dump, load
def string_labels_to_int(Y): 
  keys={}
  new_Y=[]
  for item in Y:
    if item in keys:
      new_Y.append(keys[item])
    else:
      keys.update({item:len(keys)+1})
      new_Y.append(keys[item])
  return new_Y, keys

def int_labels_to_list(Y,keys):
  new_Y=[]
  for item in Y:
    sublist=[0] * len(keys)
    sublist[item-1]=1
    sublist=torch.tensor(sublist)
    new_Y.append(sublist)
  return new_Y



class LTP:
	def __init__ (self, Xdata=None,Ydata=None, models='all'):
    if models=='all':
      self.model_list = [
        'bert-base-uncased',
        'albert-base-v2',
        'roberta-base',
        'linear_SVM',
        'multinomial_naive_bayesian',]
    elif models=='count-vectorizer':
      self.model_list = [
        'linear_SVM',
        'multinomial_naive_bayesian',]
    elif models=='cnn':
      self.model_list = [
        'bert-base-uncased',
        'albert-base-v2',
        'roberta-base',]
    else:
      print('Models not recognized, the available options are currently "all", "count-vectorizer", and "cnn")
      return

		if Xdata==Ydata==None or (Xdata==None and Ydata!=None) or (Xdata!=None and Ydata==None):
			print('Either you have not put in your own data, or you have only put in X or Y data, loading default dataset...')
			self.train_dataset_raw, self.test_dataset_raw = load_dataset('imdb', split=['train', 'test'])
			X=self.train_dataset_raw['text']+self.test_dataset_raw['text']
			Y=self.train_dataset_raw['label']+self.test_dataset_raw['label']
		else:
			X=Xdata
			Y=Ydata
			if all(isinstance(n, int) for n in Y):
				pass
			else:
				Y,keys=string_labels_to_int(Y)
				
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        stratify=Y, 
                                                        test_size=0.05,
                                                        train_size=0.05)
		#self.num_labels=len(keys)
		#self.train_dataset_raw_CNN = TensorDataset(X_train, int_labels_to_list(Y_train,keys))
		#self.test_dataset_raw_CNN = TensorDataset(X_test, int_labels_to_list(Y_test,keys))
		self.train_dataset_raw = Dataset.from_pandas(pd.DataFrame({'text':X_train, 'label': Y_train}))
		self.test_dataset_raw = Dataset.from_pandas(pd.DataFrame({'text':X_test, 'label': Y_test}))	
		self.all_metrics = {}



	def compute_metrics(self, pred):
		labels = pred.label_ids
		preds = pred.predictions.argmax(-1)
		precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
		full_report = classification_report(labels, preds, output_dict=True)
		acc = accuracy_score(labels, preds)
		return {
			'accuracy': acc,
			'f1': f1,
			'precision': precision,
			'recall': recall,
			'full_report': full_report
			}


	def get_metrics(self):
		return self.all_metrics

	def get_metrics_df(self):
		dic = self.get_metrics()
		df = pd.DataFrame.from_dict(dic)
		df = df.rename_axis("model_name", axis="columns").T
		df.reset_index(inplace=True)
		df.rename_axis()
		return df

	def print_metrics_table(self):
		dic = self.get_metrics()
		print("{:>25} {:>15} {:>15} {:>15} {:>15} {:>15}".format('Model', 'loss', 'accuracy', 'f1', 'precision', 'recall'))
		for k, v in dic.items():
			print("{:>25} {:15.5} {:15.5} {:15.5} {:15.5} {:15.5}".format(k, v['eval_loss'], v['eval_accuracy'], v['eval_f1'], v['eval_precision'], v['eval_recall']))


	def run(self):

		for model_name in self.model_list:

			training_args = TrainingArguments(
			output_dir='./results/'+model_name,
			num_train_epochs=10,
			per_device_train_batch_size=16,
			per_device_eval_batch_size=64,
			warmup_steps=500,
			weight_decay=0.01,
			#evaluate_during_training=True,
			logging_dir='./logs/'+model_name,
			)

			model = None
			tokenizer = None

			if model_name == "bert-base-uncased":
				model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)#self.num_labels)
				tokenizer = BertTokenizerFast.from_pretrained(model_name)
			elif model_name == "albert-base-v2":
				tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
				model = transformers.AlbertForSequenceClassification.from_pretrained('albert-base-v2', return_dict=True, num_labels=2)#self.num_labels)
			elif model_name == "roberta-base":
				tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
				model = transformers.RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True, num_labels=2)#self.num_labels)
			elif model_name == "linear_SVM":
				tokenizer = None
				model = 'linear_SVM'
				parameters={
				  'vect__ngram_range': [(1, 1), (1, 2)],
			  	'tfidf__use_idf': (True, False),
				  'clf__alpha': (5e-2, 1e-2,5e-3, 1e-3,5e-3),
			  	'clf__penalty': ('l2', 'l1', 'elasticnet')
				}
				classifier=SGDClassifier(loss='hinge',random_state=42,max_iter=5,tol=None)
			elif model_name == "multinomial_naive_bayesian":
				tokenizer = None
				model = 'multinomial_naive_bayesian'
				parameters= {
				  'vect__ngram_range': [(1, 1), (1, 2)],
				  'tfidf__use_idf': (True, False),
				  'clf__alpha': (1,1e-1,1e-2, 1e-3,1e-4),
				  'clf__fit_prior': (True, False),
				}
				classifier=MultinomialNB()


			if not model or not tokenizer: #use 'assert' here instead?
				print("ERROR")


			def tokenize(batch):
				return tokenizer(batch['text'], padding=True, truncation=True)

			def 


			if tokenizer is not None:

				train_dataset = self.train_dataset_raw.map(tokenize, batched=True, batch_size=len(self.train_dataset_raw))
				test_dataset = self.test_dataset_raw.map(tokenize, batched=True, batch_size=len(self.train_dataset_raw))
				train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
				test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
			else:
				train_dataset = self.train_dataset_raw
				test_dataset = self.test_dataset_raw


			if model_name== "linear_SVM" or model_name== "multinomial_naive_bayesian":
				trainer=None
				pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', classifier),
                     ])
				gs_clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1)
				gs_ind=int(len(train_dataset['label'])/10)	#use a tenth of the training dataset to do gridsearch
				gs_clf = gs_clf.fit(train_dataset['text'][:gs_ind], train_dataset['label'][:gs_ind])
				best_params=gs_clf.best_params_
				pipeline.set_params(**best_params)
				pipeline.fit(train_dataset['text'], train_dataset['label'])
				
				prediction=pipeline.predict(test_dataset['text'])
				precision, recall, f1, _ = precision_recall_fscore_support(test_dataset['label'], prediction, average=None)
				full_report=classification_report(test_dataset['label'], prediction)
				acc = accuracy_score(test_dataset['label'], prediction)
				loss=hamming_loss(test_dataset['label'], prediction)
				curr_metrics={
				'eval_loss': loss,
            			'eval_accuracy': acc,
            			'eval_f1': f1,
            			'eval_precision': precision,
            			'eval_recall': recall,
            			'eval_full_report': full_report
        }
				dump(pipeline, model_name + "_model.joblib")
				print('best parameters are:')
				print(best_params)

			else:
				trainer = Trainer(model=model,
				                      args=training_args,
				                      compute_metrics=self.compute_metrics,
				                      train_dataset=train_dataset,
				                      eval_dataset=test_dataset
				)
				trainer.train()
				curr_metrics = trainer.evaluate()
				trainer.save_model(model_name+"_model")

			self.all_metrics[model_name] = curr_metrics
			print(curr_metrics)



			# adding this fully solves the out of memory (OOM) error; https://github.com/huggingface/transformers/issues/1742
			del model, tokenizer, trainer

			# these 2 lines may not be needed
			gc.collect()
			torch.cuda.empty_cache()
	def predict(self,text):
		for model_name in self.model_list:
			if model_name == "linear_SVM" or model_name == "multinomial_naive_bayesian":
				clf = load('/content/'+model_name+'_model.joblib')
				y=clf.predict([text])
				print(y)
			else:
				if model_name == "bert-base-uncased":
					model = BertForSequenceClassification.from_pretrained('/content/bert-base-uncased_model')
					#tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased_model')
…				except:
					print('I have not got this to work yet, it was probably a good review')elf.model_list:
			if model_name == "linear_SVM" or model_name == "multinomial_naive_bayesian":
				clf = load('/content/'+model_name+'_model.joblib')
				y=clf.predict([text])
				print(y)
			else:
				if model_name == "bert-base-uncased":
					model = BertForSequenceClassification.from_pretrained('/content/bert-base-uncased_model')
					#tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased_model')
				elif model_name == "albert-base-v2":
					#tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
					model = transformers.AlbertForSequenceClassification.from_pretrained('/content/albert-base-v2_model', return_dict=True)
				elif model_name == "roberta-base":
					#tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
					model = transformers.RobertaForSequenceClassification.from_pretrained('/content/roberta-base_model', return_dict=True)
				trainer=Trainer(model=model)
				custom_text=Dataset.from_pandas(pd.DataFrame(data={'input_ids': [text]}))
				try:
					trainer.predict(test_dataset=custom_text)
				except:
					print('I have not got this to work yet, it was probably a good review')
