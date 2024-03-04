import spacy
from spacy.tokens import Doc
from spacy.tokenizer import Tokenizer
import os
import pandas as pd
import re
import pykakasi

data_fp = os.path.join("data", "cleaned", "chapters0_3_cleaned.xlsx")
df = pd.read_excel(data_fp)

nlp = spacy.load('ja_ginza_electra')

chapters_nlp = {}
for chapter_id in chapters.keys():
    chapter_sents = []
    print(f"processing chapter {chapter_id}")
    for sent in chapters[chapter_id]:
        doc = nlp(sent)
        sents = [sent for sent in doc.sents]
        chapter_sents.extend(sents)
    chapters_nlp[chapter_id] = chapter_sents

nlpp = spacy.blank(name='nlpp')
nlpp.add_pipe("ja_morphologizer", source=nlp)
for i in range(len(df)):
    first_col = df.iloc[i, 0]
    hiragana = df.iloc[i, 1]
    kanji = df.iloc[i,2]
    if type(first_col) == str and first_col.startswith('#'):
        if first_col.startswith("# new_doc_id"):
            chapter_id = first_col.split("=")[1].strip()
            if len(sent) > 0:
                doc = nlp(''.join(sent))
                for sent in doc.sents:
                    for tok in sent:
                        
                cleaned_chapters[chapter_id] = 
            sent_id = 0
            sent = []
            continue
        elif first_col.startswith("#"):
            sent_id += 1
            continue
        if type(kanji) != str:
            continue
        sent.append(kanji)
        sent.append()
        print(df.iloc[i,0])