import torch
import time
import numpy as np
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import random_split
from datasets import load_metric

torch.cuda.empty_cache()
startTime=time.time()

#initialize metric
metric = load_metric("sacrebleu")

#initializing Tokenizer & Model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50").to("cuda")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang="si_LK", tgt_lang="ta_IN")

#training data
train_data=[]
with open('data/parallel-27.04.2021-train90KDD.si-en-ta.si') as f1, open('data/parallel-27.04.2021-train90KDD.si-en-ta.ta') as f2:
    for src, tgt in zip(f1, f2):
      train_data.append(
          {
              "translation": {
                  "si_LK": src.strip(),
                  "ta_IN": tgt.strip()
              }
          }
      )
print(f'total size of train data is {len(train_data)}')

#Validationi data
valid_data=[]
with open('data/parallel-27.04.2021-tu.un.si-en-ta.si') as f1, open('data/parallel-27.04.2021-tu.un.si-en-ta.ta') as f2:
    for src, tgt in zip(f1, f2):
      valid_data.append(
          {
              "translation": {
                  "si_LK": src.strip(),
                  "ta_IN": tgt.strip()
              }
          }
      )
print(f'total size of train data is {len(valid_data)}')


#data pre-processing
def data_collator(features:list):

  inputs = [f["translation"]["si_LK"] for f in features]
  labels = [f["translation"]["ta_IN"] for f in features]  
 
  input = tokenizer(inputs, return_tensors="pt", max_length=120, padding=True, truncation=True)
  with tokenizer.as_target_tokenizer():
    label = tokenizer(labels, return_tensors="pt", max_length=120, padding=True, truncation=True).input_ids
  
  batch = input
  batch['labels'] = label

  return batch

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

#initialized model parameters
args = Seq2SeqTrainingArguments(output_dir='mbart50-ft-si-ta-run1',
                        do_train=True,
                        do_eval=True,
                        evaluation_strategy="epoch",
                        per_device_train_batch_size=16,
                        per_device_eval_batch_size=16,
                        learning_rate=2e-5,
                        weight_decay=0.01,
                        save_total_limit=24,                        
                        num_train_epochs=105,
                        predict_with_generate=True,
                        fp16=True,
                        save_steps=5000,
                        #save_strategy="epoch",
                        gradient_accumulation_steps=1,
                        eval_accumulation_steps=1,                       
                        push_to_hub=False,
                        push_to_hub_model_id='mbart50-ft-si-ta-run1')


#initlizing trainer
trainer = Seq2SeqTrainer(model=model, 
                args=args, 
                data_collator=data_collator, 
                train_dataset=train_data, 
                eval_dataset=valid_data)

tokenizer.save_pretrained('mbart50-ft-si-ta-run1')
trainer.train()
#trainer.push_to_hub()

endTime=time.time()

elapsedTime=int(endTime-startTime)

days = int(elapsedTime//86400)
hrs = int((elapsedTime%86400)//3600)
mints = int((elapsedTime%86400)%3600)

print('Time taken for training : {} days {} hrs {} min'.format(days, hrs, mints))
print('training finished')