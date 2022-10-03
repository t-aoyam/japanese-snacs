import requests
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import re
import spacy
import pandas as pd
import pykakasi
#import MeCab

URL = "https://www.aozora.gr.jp/cards/001265/files/46817_24670.html"
page = requests.get(URL)
page.encoding="Shift-JIS"

parsed = BeautifulSoup(page.text, 'lxml')
html = list(parsed.children)[2]
body = list(html.children)[3]
main = list(body.children)[4]

chapters = {}
chapter_text = []
chapter_id = ''
for item in list(main.children):
    if type(item) == Tag:
        if item.h3 is not None:
            print(chapter_text)
            print(chapter_id)
            if len(chapter_text) > 0:
                chapters[chapter_id] = chapter_text
                chapter_text = []
#            print(item.h3['class'])
            try:
                chapter_id = int(item.get_text())
            except:
                print(item.get_text())
                chapter_id = item.get_text()
        else:
            text = item.get_text().strip()
            if text != "":
                if text[0] not in ['〈', '［']:
                    text = re.sub(r"\u3000", "", text)
                    chapter_text.append(text)
    if type(item) == NavigableString:
        text = item.get_text().strip()
        if text != "":
            if text[0] not in ['〈', '［']:
                text = re.sub(r"\u3000", "", text)
                chapter_text.append(text)

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

kks = pykakasi.kakasi()

kanji = []
with open('jyouyou_list.txt', encoding='utf-8') as f:
    for line in f.readlines():
        print(line)
        kanji.append(line[0])
kanji.remove('為')
kanji.remove('御')
kanji.remove('箇')
kanji=set(kanji)

hiragana = []
with open('hiragana.txt', encoding='utf-8') as f:
    for line in f.readlines():
        hiragana.append(line[0])
hiragana=set(hiragana)

def kkc(token):
    org = token.orth_
    norm = token.norm_
    pos = token.pos_
    dep = token.dep_
    if sum([char in hiragana for char in org]) != len(org):  # if kanji is in org already
        if pos == 'NOUN':
            pass
        else:
            print('kanji in org')
            return org
    if sum([char in kanji or char in hiragana for char in norm]) != len(norm):  # if non-joyo kanji
        print('non-joyo-kanji')
        return org
    if dep == 'fixed':  # if the token is part of a fixed expression, use hiragana
        return org
    if pos in ['NOUN', 'PRON', 'ADV', 'ADJ', 'VERB']:
        if norm[-1] in hiragana:  # e.g., ちかづき -> 近づく
            if org[-1] != norm[-1]:
                return org
            else:
                return norm
        return norm
    elif pos in ['CCONJ', 'DET']:
        return org
    else:
        if org[-1] == norm[-1]:
            return norm
    return org

csv = [['Doc/Sent/Tok ID', 'Token', 'Kanji', 'Token_cor', 'XPOS', 'XPOS_cor',
        'UPOS', 'UPOS_cor', 'Target', 'SceneRole_TA', 'Function_TA', 'Notes_TA', 'Time', 'Romanized']]
#for i, chapter in enumerate(chapters_nlp[:4]):
for i in list(chapters_nlp.keys())[:4]:
    chapter = chapters_nlp[i]
    csv.append([f'# new_doc_id = chapter {i}'])
    for j, sent in enumerate(chapter):
        csv.append([f'# sent_id = {j}'])
        for k, token in enumerate(sent):
            reading = kks.convert(token.orth_)
            romanized = 'wa' if reading[0]['passport'] == 'ha' else reading[0]['passport']
            csv.append([k, token.orth_, kkc(token), '', token.tag_, '', token.pos_, '', '', '', '', '', '', romanized])
        csv.append([])
csv_df = pd.DataFrame(csv[1:], columns=csv[0])
csv_df.to_excel(r"data\chapters0_3.xlsx", index=False)
                #, encoding='utf_8_sig')



doc = nlp('いろいろかんがえてみた')
for sent in doc.sents:
    for token in sent:
        print(
            token.i,
            kkc(token),
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