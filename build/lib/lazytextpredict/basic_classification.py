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


def main():

	training_args = TrainingArguments(
		output_dir='./results',
		num_train_epochs=1,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=64,
		warmup_steps=500,
		weight_decay=0.01,
		evaluate_during_training=True,
		logging_dir='./logs',
	)

	model_list = [
	'bert-base-uncased',
	'albert-base-v2',
	'roberta-base'
	]

	train_dataset_raw, test_dataset_raw = load_dataset('imdb', split=['train[:5%]', 'test[:5%]'])


	for model_name in model_list:

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


		if not model or not tokenizer:
			print("ERROR")
			return

		def tokenize(batch):
			return tokenizer(batch['text'], padding=True, truncation=True)

		
		train_dataset = train_dataset_raw.map(tokenize, batched=True, batch_size=len(train_dataset_raw))
		test_dataset = test_dataset_raw.map(tokenize, batched=True, batch_size=len(train_dataset_raw))
		train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
		test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])



		trainer = Trainer(
		model=model,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=train_dataset,
		eval_dataset=test_dataset
		)

		trainer.train()

		print(trainer.evaluate())

		torch.cuda.empty_cache()

if __name__=="__main__":
	main()
