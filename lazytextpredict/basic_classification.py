from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from nlp import load_dataset
import torch
import numpy as np
import pandas as pd
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import transformers

def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)
	precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
	acc = accuracy_score(labels, preds)
	full_report = classification_report(labels, preds, output_dict=True)
	return {
		'accuracy': acc,
		'f1': f1,
		'precision': precision,
		'recall': recall,
		'full_report': full_report
		}


class LTP:
	def __init__ (self):
		self.model_list = [
			'bert-base-uncased',
			'albert-base-v2',
			'roberta-base'
			]
		self.train_dataset_raw, self.test_dataset_raw = load_dataset('imdb', split=['train[:1%]', 'test[:1%]'])

		self.all_metrics = {}

	def compute_metrics(self, pred):
		labels = pred.label_ids
		preds = pred.predictions.argmax(-1)
		precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
		acc = accuracy_score(labels, preds)
		return {
			'accuracy': acc,
			'f1': f1,
			'precision': precision,
			'recall': recall
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


			if not model or not tokenizer: #use 'assert' here instead?
				print("ERROR")
				return

			def tokenize(batch):
				return tokenizer(batch['text'], padding=True, truncation=True)


			train_dataset = self.train_dataset_raw.map(tokenize, batched=True, batch_size=len(self.train_dataset_raw))
			test_dataset = self.test_dataset_raw.map(tokenize, batched=True, batch_size=len(self.train_dataset_raw))
			train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
			test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])



			trainer = Trainer(
			model=model,
			args=training_args,
			compute_metrics=self.compute_metrics,
			train_dataset=train_dataset,
			eval_dataset=test_dataset
			)

			trainer.train()

			curr_metrics = trainer.evaluate()
			self.all_metrics[model_name] = curr_metrics
			print(curr_metrics)

			trainer.save_model(model_name+"_model")

			# adding this fully solves the out of memory (OOM) error; https://github.com/huggingface/transformers/issues/1742
			del model, tokenizer, trainer

			# these 2 lines may not be needed
			gc.collect()
			torch.cuda.empty_cache()
