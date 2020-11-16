from transformers import BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
from nlp import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import transformers

def compute_metrics(pred):
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


class LTP:
	def __init__ (self):
		self.model_list = [
			'bert-base-uncased',
			'albert-base-v2',
			'roberta-base'
			]
		self.train_dataset_raw, self.test_dataset_raw = load_dataset('imdb', split=['train[:5%]', 'test[:5%]'])

		#self.training_args = None

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

			print(trainer.evaluate())

			trainer.save_model(model_name+"_model")

			# adding this fully solves the out of memory (OOM) error; https://github.com/huggingface/transformers/issues/1742
			del model, tokenizer, trainer

			# these 2 lines may not be needed
			gc.collect()
			torch.cuda.empty_cache()
