import requests
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
import re
import spacy
import os
import pandas as pd
import pykakasi
from tqdm import tqdm

DATA_DIR = os.path.join('..', 'data')
RESOURCES_DIR = os.path.join(DATA_DIR, 'resources')
OUTPUT_DIR = os.path.join(DATA_DIR, 'raw')

def scrape():
    print('\n'+'='*100)
    print('scraping...'+'\n')
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
#                print(chapter_text)
#                print(chapter_id)
                if len(chapter_text) > 0:
                    chapters[chapter_id] = chapter_text
                    chapter_text = []
                    # print(item.h3['class'])
                try:
                    chapter_id = int(item.get_text())
                except:
                    # print(item.get_text())
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

    return chapters

def nlp_chapters(chapters, nlp):
    print('\n'+'='*100)
    print(r'NLP-ing the chapters...' + '\n')
    chapters_nlp = {}
    for chapter_id in tqdm(chapters.keys()):
        chapter_sents = []
        for sent in chapters[chapter_id]:
            doc = nlp(sent)
            sents = [sent for sent in doc.sents]
            chapter_sents.extend(sents)
        chapters_nlp[chapter_id] = chapter_sents
    return chapters_nlp


def get_kana_kanji():
    kanji = []
    with open(os.path.join(RESOURCES_DIR, 'jyouyou_list.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            kanji.append(line[0])
    kanji.remove('為')
    kanji.remove('御')
    kanji.remove('箇')
    kanji = set(kanji)

    hiragana = []
    with open(os.path.join(RESOURCES_DIR, 'hiragana.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            hiragana.append(line[0])
    hiragana = set(hiragana)

    return hiragana, kanji


def kkc(token, hiragana, kanji):
    org = token.orth_
    norm = token.norm_
    pos = token.pos_
    dep = token.dep_
    if sum([char in hiragana for char in org]) != len(org):  # if kanji is in org already
        if pos == 'NOUN':
            pass
        else:
#            print('kanji in org')
            return org
    if sum([char in kanji or char in hiragana for char in norm]) != len(norm):  # if non-joyo kanji
#        print('non-joyo-kanji')
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

def chapters2xlsx(start, end, chapters_nlp, hiragana, kanji, kks):
    csv = [['Doc/Sent/Tok ID', 'Token', 'Kanji', 'Token_cor', 'XPOS', 'XPOS_cor',
            'UPOS', 'UPOS_cor', 'Target', 'SceneRole_TA', 'Function_TA', 'Notes_TA', 'Time', 'Romanized']]

    #for i, chapter in enumerate(chapters_nlp[:4]):
    for i in list(chapters_nlp.keys())[25:28]:
        chapter = chapters_nlp[i]
        csv.append([f'# new_doc_id = chapter {i}'])
        for j, sent in enumerate(chapter):
            csv.append([f'# sent_id = {j}'])
            for k, token in enumerate(sent):
                reading = kks.convert(token.orth_)
                romanized = 'wa' if reading[0]['passport'] == 'ha' else reading[0]['passport']
                csv.append([k, token.orth_, kkc(token, hiragana, kanji), '', token.tag_, '', token.pos_, '', '', '', '', '', '', romanized])
            csv.append([])
    csv_df = pd.DataFrame(csv[1:], columns=csv[0])
    csv_df.to_excel(os.path.join(OUTPUT_DIR, f'chapters{start}_{end}.xlsx'), index=False)

def main():
    chapters = scrape()
    nlp = spacy.load('ja_ginza_electra')
    chapters_nlp = nlp_chapters(chapters, nlp)
    hiragana, kanji = get_kana_kanji()
    kks = pykakasi.kakasi()
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    start = 0
    print('\n'+'='*100)
    print('writing the chapters to .xlsx...' + '\n')
    for end in tqdm(range(3, 28, 3)):  # each .xlsx will be 3 chapters, just for readability
        chapters2xlsx(start, end, chapters_nlp, hiragana, kanji, kks)
        start = end+1

if __name__ == '__main__':
    main()