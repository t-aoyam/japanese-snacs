
import conllu
import re
from collections import defaultdict
import pandas as pd
import numpy as np

file_dev = "ja_gsd-ud-dev.conllu"
file_test = "ja_gsd-ud-test.conllu"
file_train = "ja_gsd-ud-train.conllu"

with open(file_dev, encoding='utf-8') as f:
    dev = f.read()

with open(file_train, encoding='utf-8') as f:
    train = f.read()

with open(file_test, encoding='utf-8') as f:
    test = f.read()

jud = train+dev+test
jud_list = conllu.parse(jud)


upos2xpos = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0)))
for i, sent in enumerate(jud_list):
    print(f"{i}/{len(jud_list)}")
    for token in sent:
        upos2xpos[token['upos']][token['xpos']][token['lemma']] += 1

xpos2upos = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda:0)))
for i, sent in enumerate(jud_list):
    print(f"{i}/{len(jud_list)}")
    for token in sent:
        xpos2upos[token['xpos']][token['upos']][token['lemma']] += 1

for xpos in xpos2upos:
    total = 0
    for upos in xpos2upos[xpos]:
        for lemma in xpos2upos[xpos][upos]:
            total += xpos2upos[xpos][upos][lemma]
    xpos2upos[xpos]['total'] = total

table = []
for xpos in xpos2upos:
    if '助詞' not in xpos:
        continue
    for upos in xpos2upos[xpos]:
        if upos == 'total':
            continue
        row = [xpos, upos]
        lemmas = []
#        print(xpos2upos[xpos][upos])
        for lemma in xpos2upos[xpos][upos]:
            lemmas.append(f"{lemma} ({xpos2upos[xpos][upos][lemma]})")
        lemmas_joined = ", ".join(lemmas)
        row.append(lemmas_joined)
        table.append(row)

df = pd.DataFrame(table, index = [row[0] for row in table], columns=['XPOS', 'UPOS', 'lemmas'])
df2 = df.loc[:, ['UPOS', 'lemmas']]
df2.style.to_latex("table1.tex", encoding="UTF-8")
''.join([item['form'] for item in jud_list[0]])


def lookup(sent_num=None, token=None,
           xpos_q=None, upos_q=None, lemma_q=None):
    sent_num_r = sent_num
    token_r = token

    def randomize(sent_num_r, token_r,
                  sent_num=sent_num, token=token):        
        if sent_num == None:
            sent_num_r = np.random.randint(0, 8100)
        if token == None:
            token_r = np.random.randint(0, len(jud_list[sent_num_r]))
        xpos = jud_list[sent_num_r][token_r]['xpos']
        upos = jud_list[sent_num_r][token_r]['upos']
        lemma = jud_list[sent_num_r][token_r]['lemma']
        return (sent_num_r, token_r, xpos, upos, lemma)
    
    def is_xpos(sent_num_r, token_r,
               xpos, upos, lemma):
        is_xpos = xpos_q == xpos
        return is_xpos
    
    def is_upos(sent_num_r, token_r,
               xpos, upos, lemma):
        is_upos = upos_q == upos
        return is_upos
    
    def is_lemma(sent_num_r, token_r,
               xpos, upos, lemma):
        is_lemma = lemma_q == lemma
        return is_lemma

    sent_num_r, token_r, xpos, upos, lemma = randomize(sent_num_r, token_r)
    
    cond_func = {xpos_q:is_xpos, upos_q:is_upos, lemma_q:is_lemma}
    conds = [cond for cond in [xpos_q, upos_q, lemma_q] if cond not in [None, False]]
    print(conds)
    while sum([cond_func[cond](sent_num_r, token_r, xpos, upos, lemma) for cond in conds]) != len(conds):
        sent_num_r, token_r, xpos, upos, lemma = randomize(sent_num_r, token_r)
#        print(f"satisfied: {sum([cond_func[cond](match_type, layer_r, sent_num_r, token_r, gold_word,pred_word, gold_pos, pred_pos) for cond in conds])}")
#        print(f"necessary: {len(conds)}")

#generate predictions on the specified sentence and token
    sent = ''.join([item['form'] for item in jud_list[sent_num_r]])
    return ((xpos, upos, lemma), sent, (sent_num_r, token_r))

lookup(xpos_q = '助詞-接続助詞', lemma_q='で', upos_q='SCONJ')


for i, tok in enumerate(jud_list[6349]):
    print(jud_list[6349][i])
jud_list[6349][20]

jud_list[4788][0]

jud_list[6336][9]

jud_list[2697][6]
