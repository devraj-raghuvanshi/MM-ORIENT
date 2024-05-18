!pip install transformers

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, BertModel

msg=pd.read_csv("/path/to/memotion/csv")

text_training=msg["caption"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

inputs = tokenizer(text_testing[0], return_tensors="pt",truncation=True,max_length=128,padding="max_length")
outputs = model(**inputs)
bert_test=outputs.last_hidden_state.detach().numpy()

for i in range(1,len(text_testing)):
  inputs = tokenizer(text_testing[i], return_tensors="pt",truncation=True,max_length=128,padding="max_length")
  outputs = model(**inputs)
  bert_test=np.append(bert_test,outputs.last_hidden_state.detach().numpy(),axis=0)
  print(i)

np.save("path/to/save/features",bert_test)