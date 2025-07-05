!pip install transformers

import pandas as pd
import numpy as np
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor

df = pd.read_csv("/path/to/memotion/csv")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

testingfile=df["image_name"].to_numpy()

testingtext=df["caption"].to_numpy()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

folder="/path/to/image/directory/"

image = Image.open(folder+testingfile[0])
inputs = processor(text=testingtext[0], images=image, return_tensors="pt", padding=True,truncation=True)
outputs = model(**inputs)
img_embed=outputs["image_embeds"].detach().numpy()
text_embed=outputs["text_embeds"].detach().numpy()

for i in range(1,len(testingfile)):
  image = Image.open(folder+testingfile[i])
  inputs = processor(text=testingtext[i], images=image, return_tensors="pt", padding=True,truncation=True)
  outputs = model(**inputs)
  img=outputs["image_embeds"].detach().numpy()
  img_embed=np.append(img_embed,img,axis=0)
  txt=outputs["text_embeds"].detach().numpy()
  text_embed=np.append(text_embed,txt,axis=0)

np.save("path/to/save/text/features",text_embed)
np.save("path/to/save/image/features",img_embed)