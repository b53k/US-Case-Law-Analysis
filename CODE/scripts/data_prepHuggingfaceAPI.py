import pandas as pd
import os
from datasets import load_dataset
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

script_dir = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.dirname(script_dir)
data_loc = os.path.join(cwd, 'Processed Data')

dataset = load_dataset("free-law/Caselaw_Access_Project", split = 'train', streaming = True)  # From HuggingFace

data_list = []
for idx, example in enumerate(dataset):
    data_list.append(example)
    if idx >= 20000:  # collecting first 3000 examples
        break

df = pd.DataFrame(data_list)
print(df.head())

print(df['text'][1])
df['Conclusion'] = df['text'].str.extract(r"CONCLUSION\s+(.*)", flags=re.DOTALL, expand=True)
df = df.dropna(subset=['Conclusion']).reset_index(drop=True)
print(df['Conclusion'][1])
def extract_next_sentences(text, header="FACTS", num_sentences=4):
    # Locate the header and capture all text following it
    pattern = re.compile(rf"{header}\s*(.*)", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if not match:
        return ""
    section_text = match.group(1)
    # Tokenize the captured text into sentences
    sentences = sent_tokenize(section_text)
    # Join the first 'num_sentences' sentences (if available)
    return " ".join(sentences[:num_sentences])

# Apply the function to the 'text' column and create a new column 'Next4Sentences'
df['Next4Sentences'] = df['text'].apply(lambda x: extract_next_sentences(x, header="FACTS", num_sentences=4))
print(df['Next4Sentences'][1])



##### Labeling the outcome ######
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.eval()

def predict_winner(conclusion_text):
    if not conclusion_text or pd.isna(conclusion_text):
        return None
    inputs = tokenizer(conclusion_text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    pred_label = torch.argmax(outputs.logits, dim=1).item()
    return "appellant" if pred_label == 0 else "appellee"

# Combine Conclusion and Next4Sentences (fill NaNs with empty string)
df['CombinedOutcomeText'] = df['Conclusion'].fillna('') + " " + df['Next4Sentences'].fillna('')

df['winner'] = df['CombinedOutcomeText'].apply(predict_winner)
print(df[['Conclusion', 'winner']].head())
df.to_csv('caseLawRaw.csv', index=False)
