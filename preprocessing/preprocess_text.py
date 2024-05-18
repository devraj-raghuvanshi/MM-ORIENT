import numpy as np
import pandas as pd
import re
import string

df_concat_new = pd.read_csv("/path/to/memotion/csv/containing/rephrased/text")

df_concat_new = df_concat.drop_duplicates(subset='rephrased_text', keep='first')    # rephrased_text is the column containing text rephrased using gpt-3.5-turbo

def preprocess_ocr_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove non-alphanumeric characters and extra whitespaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text

df_concat_new = df_concat_new['rephrased_text'].apply(preprocess_ocr_text)

def contains_sequence(sent):
    sequences = ['www.', '.com', 'website', 'Sentence']
    for sequence in sequences:
        if sequence in sent:
            return 1
    return 0

clean = df_concat_new['rephrased_text'].apply(contains_sequence)

df_concat_final = df_concat_new.loc[clean].reset_index(drop=True)

def remove_from_starting(input_string):
    if input_string.startswith("- "):
        return True, input_string[2:]
    elif input_string.startswith("• "):
        return True, input_string[2:]
    return False, input_string

df_concat_final['rephrased_text'] = df_concat_final['rephrased_text'].apply(lambda x: remove_from_starting(x)[1])

def remove_words_starting_with_at_symbol(input_string):
    words = input_string.split()
    new_words = []
    at_present = False

    for word in words:
        if word.startswith("@"):
            at_present = True
        else:
            new_words.append(word)

    return at_present, " ".join(new_words)

df_concat_final = df_concat_final.loc[df_concat_final['rephrased_text'].apply(lambda x: not remove_words_starting_with_at_symbol(x)[0])].reset_index(drop=True)

df_concat_final.to_csv("/path/to/save/output/csv")