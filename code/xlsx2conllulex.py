import os
import pandas as pd
import pykakasi

# get all the spreadsheets into one big DataFrame
data_fps = os.listdir(os.path.join('data', 'cleaned'))
data_fps.sort(key = lambda x: int(x.split('_')[1]))
df = pd.concat([pd.read_excel(os.path.join('data', 'cleaned', data_fp)) for data_fp in data_fps],
               ignore_index=True)

# putting SS into a correct capitalization
capitalize_ss_fp = os.path.join("data", "supersenses_capitalization.tab")
with open(capitalize_ss_fp) as f:
    capitalize_ss = {lower:cap for lower, cap in [item.strip().split('\t') for item in f.readlines()]}
kks = pykakasi.kakasi()

def get_chapter_sent_tok(df):
    chapter_id = -1  # chapter ID start from 0 (foreword)
    sent_id = 0  # sent ID start from 1
    tok_id = 0  # tok ID start from 1
    BIO = "O"  # keep track of MWE status
    chapters = dict()
    sent = []
    for i in range(len(df)):
        first_col = df.iloc[i, 0]
        if type(first_col) == str:
            if first_col.startswith("# new"):
                if len(sent) > 0:  # last sentence of the previous chapter
                    chapters[chapter_id][sent_id]['text'] = sent
                    sent = []
                chapter_id += 1
                sent_id = 0  # reset the sentence count
                chapters[chapter_id] = dict()
            elif first_col.startswith("#"):
                if len(sent) > 0:
                    chapters[chapter_id][sent_id]['text'] = sent
                    sent = []
                sent_id += 1
                tok_id = 0  # reset the tok count
                chapters[chapter_id][sent_id] = {'text': None,
                                                 'toks': dict()}
            continue
        tok = df.loc[i, "Token"]
        kanji = df.loc[i, "Kanji"]
        if BIO != 'O':  # if previously part of MWE
            BIO = 'I' if type(df.loc[i, "MWE"]) == str else "O"
        else:
            BIO = 'B' if type(df.loc[i, "MWE"]) == str else "O"            
        if type(tok) == str:
            sr = 'p.' + capitalize_ss[df.loc[i, "SR"].lower()] if\
                type(df.loc[i, "SR"]) == str else '_'
            f = 'p.' + capitalize_ss[df.loc[i, "F"].lower()] if\
                type(df.loc[i, "F"]) == str else '_'
            sent.append(tok)
            tok_id += 1
            rom = ''.join([converted['passport'] for converted in kks.convert(tok)])
            lexcat = f"{BIO}-P-{sr}|{f}" if sr != '_' else "O-X"  # TODO: add POS later
            lexcat = "I_" if BIO == 'I' else lexcat
            # [id, tok, lemma, xpos, upos, morph, head, dep, ?, ?, ?, eupos?,
            # lemma, sr, f, ?, ?, ?, lexcat]
            tok_info = [tok_id, tok, rom, kanji, '_', '_', '_', '_', '_', '_', '_', '_',
                        rom, sr, f, '_', '_', '_', lexcat]
            chapters[chapter_id][sent_id]['toks'][tok_id] = tok_info
    if len(sent) > 0:  # one very last sentence!
        chapters[chapter_id][sent_id]['text'] = sent
    return chapters

def dct2conllulex(dct):
    conllulex = []
    for cid in dct:
        conllulex.append(f"# newdoc id = lpp_jp-{cid}")
        for sid in dct[cid]:
            conllulex.extend([f"# sent_id = lpp_jp.{cid}.{sid}",
                              f"# text = {' '.join(dct[cid][sid]['text'])}"])
            for tid in dct[cid][sid]['toks']:
                conllulex.append('\t'.join([str(col) for col in dct[cid][sid]['toks'][tid]]))
    return conllulex
chapters = get_chapter_sent_tok(df)
conllulex = dct2conllulex(chapters)
with open(os.path.join('data', 'lpp_jp.conllulex'),
          'w', newline='\n') as f:
    f.write('\n'.join(conllulex))