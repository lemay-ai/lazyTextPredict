import pandas as pd
import gc
import transformers
from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from nlp import load_dataset, Dataset
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from random import sample, choices

class LTP:
	def __init__ (self):
		self.model_list = [
			'linear_SVM',
			'multinomial_naive_bayesian',
			'bert-base-uncased',
			'albert-base-v2',
			'roberta-base'

			]
		self.train_dataset_raw, self.test_dataset_raw = load_dataset('imdb', split=['train', 'test'])
		self.train_dataset_raw_random=choices(self.train_dataset_raw,k=int(len(self.train_dataset_raw)/100))
		self.test_dataset_raw_random=choices(self.test_dataset_raw,k=int(len(self.test_dataset_raw)/100))
		self.train_dataset_raw = Dataset.from_pandas(pd.DataFrame(self.train_dataset_raw_random))
		self.test_dataset_raw = Dataset.from_pandas(pd.DataFrame(self.test_dataset_raw_random))
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
			num_train_epochs=1,
			per_device_train_batch_size=16,
			per_device_eval_batch_size=64,
			warmup_steps=500,
			weight_decay=0.01,
			evaluate_during_training=True,
			logging_dir='./logs/'+model_name,
			)

			model = None
			tokenizer = None

			if model_name == "bert-base-uncased":
				model = BertForSequenceClassification.from_pretrained(model_name)
				tokenizer = BertTokenizerFast.from_pretrained(model_name)
			elif model_name == "albert-base-v2":
				tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v2')
				model = transformers.AlbertForSequenceClassification.from_pretrained('albert-base-v2', return_dict=True)
			elif model_name == "roberta-base":
				tokenizer = transformers.RobertaTokenizer.from_pretrained('roberta-base')
				model = transformers.RobertaForSequenceClassification.from_pretrained('roberta-base', return_dict=True)
			elif model_name == "linear_SVM":
				tokenizer = None
				model = 'linear_SVM'
				parameters=parameters = {
				  'vect__ngram_range': [(1, 1), (1, 2)],
			  	'tfidf__use_idf': (True, False),
				  'clf__alpha': (5e-2, 1e-2,5e-3, 1e-3,5e-3),
			  	'clf__penalty': ('l2', 'l1', 'elasticnet')
				}
				classifier=SGDClassifier(loss='hinge',random_state=42,max_iter=5,tol=None)
			elif model_name == "multinomial_naive_bayesian":
				tokenizer = None
				model = 'multinomial_naive_bayesian'
				parameters=parameters = {
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
				pipeline.fit(train_dataset['text'], train_dataset['label'])
				best_params=gs_clf.best_params_
				pipeline.set_params(**best_params)
				prediction=pipeline.predict(test_dataset['text'])
				precision, recall, f1, _ = precision_recall_fscore_support(test_dataset['label'], prediction, average='binary')
				full_report=classification_report(test_dataset['label'], prediction)
				acc = accuracy_score(test_dataset['label'], prediction)
				curr_metrics={
            			'accuracy': acc,
            			'f1': f1,
            			'precision': precision,
            			'recall': recall,
            			'full_report': full_report
        }
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
