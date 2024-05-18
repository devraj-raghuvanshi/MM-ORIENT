!pip install openai

import numpy as np
import pandas as pd
import requests
import re
import time
import csv

df = pd.read_csv("path/to/dataframe")

def get_chat_completion(text):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer WRITE_YOUR_KEY_HERE"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful, pattern-following assitant that removes the URLs and usernames from sentece, and then rephrases the sentence in 5 different forms."},
            {"role": "system", "name": "example_user", "content": "you mean to tell me spoons don't actually sound like airplanes? www.abc.com"},
            {"role": "system", "name": "example_assistant", "content": "Correct me if I'm wrong, but spoons don't sound like airplanes, do they?\nI had no idea that spoons don't actually make airplane-like sounds.\nCan you clarify if spoons don't sound like airplanes, please?\nIt's news to me that spoons don't actually sound like airplanes.\nSo, the idea that spoons sound like airplanes is not accurate, is it?"},
            {"role": "system", "name": "example_user", "content": "me: i should keep my identity a secret me to me: but has he heard about the tragedy of darth plagueis the wise? facebook.com"},
            {"role": "system", "name": "example_assistant", "content": "While conversing with myself, I question whether he's been informed about Darth Plagueis the Wise's tragedy.\nIt occurs to me that I should hide my true identity from others.\nReflecting on my situation, I contemplate whether he's acquainted with the tragedy of Darth Plagueis the Wise.\nPersonally, I believe in the importance of preserving my anonymity.\nIn my inner dialogue, I wonder if he's ever heard the tragic tale of Darth Plagueis the Wise."},
            {"role": "user", "content": text}
            ],
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code)
        print(response.text)
        return None

def extract_bullet_points(text):
    bullet_points = re.findall(r'\d+\.\s(.+)', text)
    return bullet_points

def extract_lines(input_string):
    lines_list = input_string.split("\n")
    return [item for item in lines_list if item.strip()]
    return lines_list

csv_output_file = "path/to/outputfile"

column_names = ['i', 'orig_ind', 'image_name', 'original_text', 'rephrased_text', 'humour', 'sarcasm', 'offensive', 'motivational', 'overall_sentiment']                              #change
with open(csv_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(column_names)

not_done = []

WAIT_TIME_SECONDS = 1

for i, sent in enumerate(df['text_corrected']):
  result = get_chat_completion('paraphrase this entire sentence into 5 different sentences: ' + sent.lower())
  if result:
      res = result['choices'][0]['message']['content']

      if len(extract_bullet_points(res)) == 5:
        rephrased_sentences = extract_bullet_points(res)

      elif len(extract_lines(res)) == 5:
        rephrased_sentences = extract_lines(res)

      else:
        not_done.append(i)
        print(i, '------------------- NOT DONE -------------------')
        continue

      with open(csv_output_file, mode='a', newline='') as csv_file:
          writer = csv.writer(csv_file)
          for rephr_sent in rephrased_sentences:
            writer.writerow([df['index'][i], i, df['image_name'][i], sent, rephr_sent, df['humour'][i], df['sarcasm'][i], df['offensive'][i], df['motivational'][i], df['overall_sentiment'][i]])

      print(i, 'DONE')

      # time.sleep(WAIT_TIME_SECONDS)

