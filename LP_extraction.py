from pdfminer.high_level import extract_text, extract_pages
import pdfminer
import spacy
import re
import pandas as pd
import pykakasi

file = "C:\\Users\\aozsa\\google Drive\\Georgetown\\Spring 2022\\LING 672\\Final Project\\The_Little_Prince.pdf"
from pdfminer.high_level import extract_pages
for page_layout in extract_pages(file):
    page = []
    for element in page_layout:
        page.append(element)
    elements.append(page)

raw_text = ''
for page in elements:
    for element in page:
        if element.y1 < 100:
            continue
        else:
            if isinstance(element, (pdfminer.layout.LTTextBoxHorizontal,
                                     pdfminer.layout.LTTextBox)):
                raw_text += element.get_text()

chapters = {}
text = ''
for i, line in enumerate(raw_text.split("\n")):
    if line.strip().isdigit():
        chapter = int(line.strip())-1
        chapters[chapter] = text
        text = ''
    else:
        text += line
    if i == len(raw_text.split("\n"))-1:
        chapters[int(list(chapters.keys())[-1])+1] = text

nlp = spacy.load('ja_ginza_electra')

chapters_nlp = {}
for chapter in chapters.keys():
    print(f"processing chapter {chapter}")
    doc = nlp(re.sub(' ', '', chapters[chapter]))
    sents = [sent for sent in doc.sents]
    chapters_nlp[chapter] = sents

kks = pykakasi.kakasi()
csv = [['Doc/Sent/Tok ID', 'Token', 'Token_cor', 'XPOS', 'XPOS_cor',
        'UPOS', 'UPOS_cor', 'Target', 'SceneRole_TA', 'Function_TA', 'Notes_TA', 'Time', 'Romanized']]
#for i, chapter in enumerate(chapters_nlp[:4]):
for i in list(chapters_nlp.keys())[:4]:
    chapter = chapters_nlp[i]
    csv.append([f'# new_doc_id = chapter {i}'])
    for j, sent in enumerate(chapter):
        csv.append([f'# sent_id = {j}'])
        for k, token in enumerate(sent):
            reading = kks.convert(token.orth_)
            csv.append([k, token.orth_, '', token.tag_, '', token.pos_, '', '', '', '', '', '', reading[0]['passport']])
        csv.append([])
csv_df = pd.DataFrame(csv[1:], columns=csv[0])
#csv_df.to_csv('Ch0_3_0424.csv', index=False, encoding='utf_8_sig')
csv_df.to_excel('Ch0_3_0506.xlsx', index=False, encoding='utf_8_sig')
#csv_df.to_csv('Ch0_3_0424.csv', index=False, encoding='ANSI')

doc = nlp(text)
for sent in doc.sents:
    for token in sent:
        print(
            token.i,
            token.orth_,
            token.lemma_,
            token.norm_,
            token.morph.get("Reading"),
            token.pos_,
            token.morph.get("Inflection"),
            token.tag_,
            token.dep_,
            token.head.i,
        )
    print('EOS')