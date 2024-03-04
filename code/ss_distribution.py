import os
import xml.etree.ElementTree as ET
import re
import pykakasi
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import json
from numpy import dot
from numpy.linalg import norm
import torch
kks = pykakasi.kakasi()

capitalize_ss_fp = os.path.join("data", "supersenses_capitalization.tab")
with open(capitalize_ss_fp) as f:
    capitalize_ss = {lower:cap for lower, cap in [item.strip().split('\t') for item in f.readlines()]}

# get English distribution from English Little Prince
"""
fp = os.path.join('data', 'prince_en_without_1_4_5.conllulex')
p_en = dict()
with open(fp) as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("#") or len(line) == 0:
            continue
        line = line.split("\t")
        lemma = line[2]
        sr = line[13]
        f = line[14]
        lexcat = line[18]
        if lexcat == "_"\
            or len(lexcat.split('-')) < 2\
                or lexcat.split('-')[1] != 'P':
            continue
        if lemma not in p_en:
            p_en[lemma] = {'sr': {ss: 0 for ss in sorted(list(capitalize_ss.values()))},
                           'f': {ss: 0 for ss in sorted(list(capitalize_ss.values()))}}
        sr = sr.split('.')[1]
        f = f.split('.')[1]
        p_en[lemma]['sr'][sr] += 1
        p_en[lemma]['f'][f] += 1
"""
def _add_p(dct, p, sr, f):
    if p not in dct:
        dct[p] = {'sr': {ss: 0 for ss in sorted(list(capitalize_ss.values()))},
                       'f': {ss: 0 for ss in sorted(list(capitalize_ss.values()))}}
    dct[p]['sr'][sr] += 1
    dct[p]['f'][f] += 1
    return dct

fp = os.path.join('data', 'prince_en_without_1_4_5.conllulex')
p_en = dict()
p = None
with open(fp) as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("#") or len(line) == 0:
            if line.startswith("# text"):
                sent = line.split('=')[1].strip()
            continue
        line = line.split("\t")
        lexcat = line[-1]
        lemma = line[2]
        BIO = lexcat.split('-')[0]

        if lexcat == "_"\
            or (line[13].split('.')[0] != 'p' and BIO != "I_"):
                if type(p) == list:  # MWE ended last line
                    p_en = _add_p(p_en, ' '.join(p), sr, f)
                p = None
                continue

        if BIO == "B":  # MWE begins
            if type(p) == list:  # MWE ended last line
                p_en = _add_p(p_en, ' '.join(p), sr, f)
            sr = line[13].split('.')[1]
            f = line[14].split('.')[1]
            p = [lemma]
        
        elif BIO == "I_" and type(p) == list:  # inside MWE
            print('shit')
            p.append(lemma)
            
        elif BIO in ["O", "I~"]:  # single word expression or MWE but of different kind (belong *to*)
            if type(p) == list:  # MWE ended last line
                p_en = _add_p(p_en, ' '.join(p), sr, f)
            sr = line[13].split('.')[1]
            f = line[14].split('.')[1]
            p = lemma
            p_en = _add_p(p_en, p, sr, f)


# TODO: get J distribution and compare!

data_fp1 = os.path.join("data", "cleaned", "chapters0_3_cleaned.xlsx")
data_fp2 = os.path.join("data", "cleaned", "chapters4_6_cleaned.xlsx")
data_fp3 = os.path.join("data", "cleaned", "chapters7_9_cleaned.xlsx")
data_fp4 = os.path.join("data", "cleaned", "chapters10_12_cleaned.xlsx")
df1 = pd.read_excel(data_fp1)
df2 = pd.read_excel(data_fp2)
df3 = pd.read_excel(data_fp3)
df4 = pd.read_excel(data_fp4)

df = pd.concat([df1, df2, df3, df4], ignore_index=True)


p_jp = dict()
sr_f_list = []
sr_list = []
f_list = []
num_tok = 0
num_sents = 0
num_chaps = 0
for i in range(len(df)):
    first_col = df.iloc[i, 0]
    if type(first_col) == str:
        if first_col.startswith("# new"):
            num_chaps += 1
        elif first_col.startswith("#"):
            num_sents += 1
#    print(i)
    tok = df.loc[i, "Token"]
    mwe = df.loc[i, "Token-MWE"]

    if type(tok) == str:
        num_tok += 1
    if type(tok) == str and type(df.loc[i, "SR"]) == str:
#        tok = df.loc[i, "Token"] + ' (' + df.loc[i, "Romanized"] + ')'
        rom = ''.join([converted['passport'] for converted in kks.convert(tok)])
        if type(mwe) == str:
            rom = ''.join([converted['passport'] for converted in kks.convert(mwe)])
            print(rom)
        sr = capitalize_ss[df.loc[i, "SR"].lower()]
        f = capitalize_ss[df.loc[i, "F"].lower()]
        if type(sr) == str and type(f) == str:
            sr_f = "_".join([sr, f])
            sr_f_list.append(sr_f)
            sr_list.append(sr)
            f_list.append(f)
            if rom not in p_jp:
                p_jp[rom] = {'sr': {ss: 0 for ss in sorted(list(capitalize_ss.values()))},
                             'f': {ss: 0 for ss in sorted(list(capitalize_ss.values()))},
                             'sr_f': {ss: 0 for ss in sorted(list(capitalize_ss.values()))}}
            if sr not in p_jp[rom]['sr']:
                p_jp[rom]['sr'][sr] = 0
            if f not in p_jp[rom]['f']:
                p_jp[rom]['f'][f] = 0
            if sr_f not in p_jp[rom]['sr_f']:
                p_jp[rom]['sr_f'][sr_f] = 0
            p_jp[rom]['sr'][sr] += 1
            p_jp[rom]['f'][f] += 1
            p_jp[rom]['sr_f'][sr_f] += 1

# convert both distributions to relative frequencies
p_jp_rel = dict()
for p in p_jp:
    p_jp_rel[p] = dict()
    p_jp_rel[p]['sr'] = [(sr, freq/sum(p_jp[p]['sr'].values()))\
                         for sr, freq in p_jp[p]['sr'].items()]
    p_jp_rel[p]['f'] = [(f, freq/sum(p_jp[p]['f'].values()))\
                        for f, freq in p_jp[p]['f'].items()]

p_en_rel = dict()
for p in p_en:
    p_en_rel[p] = dict()
    p_en_rel[p]['sr'] = [(sr, freq/sum(p_en[p]['sr'].values()))\
                         for sr, freq in p_en[p]['sr'].items()]
    p_en_rel[p]['f'] = [(f, freq/sum(p_en[p]['f'].values()))\
                        for f, freq in p_en[p]['f'].items()]

# compute JS divergence

def measure_divergence(ss_dist1, ss_dist2):
    values1 = [item[1] for item in ss_dist1]
    keys1 = [item[0] for item in ss_dist1]
    values2 = [item[1] for item in ss_dist2]
    keys2 = [item[0] for item in ss_dist2]
    assert keys1 == keys2  # ensuring that the order is the same
    return jensenshannon(values1, values2)

measure_divergence(p_en_rel['of']['f'], p_jp_rel['no']['f'])
measure_divergence(p_jp_rel['no']['f'], p_en_rel['of']['f'])

# compute within-lang divergence

def create_all_combo(p_list1, p_list2=None):
    if p_list2:
        num_all_perm = len(p_list1) * (len(p_list2))        
        all_combo = np.array(
            [[[p1, p2] for p1 in p_list1]\
                              for p2 in p_list2]
                ).reshape((num_all_perm, 2))
        all_combo = [[str(item1), str(item2)] for item1, item2 in all_combo]
    else:
        p_list2 = p_list1
        num_all_perm = len(p_list1) * (len(p_list1) - 1)
        all_combo = np.array(
            [[sorted([p1, p2]) for p1 in p_list1 if p1 != p2]\
                              for p2 in p_list2]
                ).reshape((num_all_perm, 2))
        all_combo = list(set(['@'.join(list(combo)) for combo in all_combo]))
        assert len(all_combo) == (num_all_perm / 2)
        all_combo = sorted([combo.split('@') for combo in all_combo],
                       key=lambda x: (x[0], x[1]))
    return all_combo

def create_ss_ranking(all_combo, ss_dict1, ss_dict2, sr=False):
    srf = 'f'
    if sr:
        srf = 'sr'
    ranking = []
    for combo in tqdm(all_combo):
        dist1 = ss_dict1[combo[0]][srf]
        dist2 = ss_dict2[combo[1]][srf]
        divergence = measure_divergence(dist1, dist2)
        ranking.append([combo, divergence])
    return sorted(ranking, key=lambda x: x[1])

def create_embed_ranking(all_combo, p2ss2emb1, p2ss2emb2=None,
                         sr=False, layer=12):
    srf = 'f'
    if sr:
        srf = 'sr'
    if p2ss2emb2 is None:
        p2ss2emb2 = p2ss2emb1
    ranking = []
    for combo in tqdm(all_combo):

        # get mean embedding for all occurrences for word1
        embeds1 = []
        for ss in p2ss2emb1[str(layer)][combo[0]][srf]:
            embeds1.extend(
                [torch.tensor(embed) for embed in\
                 p2ss2emb1[str(layer)][combo[0]][srf][ss]
                 ]
                    )
        mean_embed1 = torch.stack(embeds1).mean(dim=0)

        # get mean embedding for all occurrences for word2
        embeds2 = []
        for ss in p2ss2emb2[str(layer)][combo[1]][srf]:
            embeds2.extend(
                [torch.tensor(embed) for embed in\
                 p2ss2emb2[str(layer)][combo[1]][srf][ss]
                 ]
                    )
        mean_embed2 = torch.stack(embeds2).mean(dim=0)

        distance = dot(mean_embed1, mean_embed2)/(norm(mean_embed1)*norm(mean_embed2))
        ranking.append([combo, distance])
    return sorted(ranking, key=lambda x: x[1])

def pearson_ss_embed(ss_divergence, embed_distance):
    ss_divergence.sort(key=lambda x:x[0])
    embed_distance.sort(key=lambda x:x[0])
    assert [item[0] for item in ss_divergence] == [item[0] for item in embed_distance]
    return pearsonr([item[1] for item in ss_divergence],
                    [item[1] for item in embed_distance])

"""English"""
## compute ss distribution divergence
all_combo_en = create_all_combo(list(p_en.keys()))
ranking_en = create_ss_ranking(all_combo_en,
                               p_en_rel, p_en_rel, sr=True)

# compute embedding distance in monolingual BERT
with open(os.path.join('data', 'p2ss2emb_en_mbert.json')) as f:
    p2ss2emb_en = json.loads(f.read())

# calculate embedding distance
ranking_embed_en = create_embed_ranking(all_combo_en, p2ss2emb_en,
                                        sr=True, layer=12)

# pearson correlation between ss divergence and embedding distance
pearson_ss_embed(ranking_en, ranking_embed_en)

# by layers
pearson_layer_en = []
for i in range(1,13):
    ranking_embed_en = create_embed_ranking(all_combo_en, p2ss2emb_en,
                                            sr=True, layer=i)
    pearson_layer_en.append(
        [i, pearson_ss_embed(ranking_en, ranking_embed_en)]
        )



"""Japanese"""
## compute ss distribution divergence
all_combo_jp = create_all_combo(list(p_jp.keys()))
ranking_jp = create_ss_ranking(all_combo_jp,
                               p_jp_rel, p_jp_rel, sr=True)


## compute embedding distance in monolingual BERT
with open(os.path.join('data', 'p2ss2emb_jp_mbert.json')) as f:
    p2ss2emb_jp = json.loads(f.read())

## needs some postprocessing to change from kana to romanized for J
ps = list(p2ss2emb_jp['1'].keys())
kana2rom = {p: ''.join([item['passport'] for item in kks.convert(re.sub(' ', '', p))])
            for p in ps}
for layer in p2ss2emb_jp:
    for kana in ps:
        p2ss2emb_jp[layer][kana2rom[kana]] = p2ss2emb_jp[layer][kana]
        del p2ss2emb_jp[layer][kana]

## calculate embedding distance
ranking_embed_jp = create_embed_ranking(all_combo_jp, p2ss2emb_jp,
                                        sr=True, layer=12)

# pearson correlation between ss divergence and embedding distance
pearson_ss_embed(ranking_jp, ranking_embed_jp)

# by layer
pearson_layer_jp = []
for i in range(1,13):
    ranking_embed_jp = create_embed_ranking(all_combo_jp, p2ss2emb_jp,
                                            sr=True, layer=i)
    pearson_layer_jp.append(
        [i, pearson_ss_embed(ranking_jp, ranking_embed_jp)]
        )


"""CROSS LINGUAL"""
## compute ss distribution divergence
all_combo_en_jp = create_all_combo(list(p_en.keys()), list(p_jp.keys()))
ranking_en_jp = create_ss_ranking(all_combo_en_jp,
                                  p_en_rel, p_jp_rel, sr=True)


## compute embedding distance in monolingual BERT
with open(os.path.join('data', 'p2ss2emb_en_mbert.json')) as f:
    p2ss2emb_en = json.loads(f.read())
with open(os.path.join('data', 'p2ss2emb_jp_mbert.json')) as f:
    p2ss2emb_jp = json.loads(f.read())

## needs some postprocessing to change from kana to romanized for J
ps = list(p2ss2emb_jp['1'].keys())
kana2rom = {p: ''.join([item['passport'] for item in kks.convert(re.sub(' ', '', p))])
            for p in ps}
for layer in p2ss2emb_jp:
    for kana in ps:
        p2ss2emb_jp[layer][kana2rom[kana]] = p2ss2emb_jp[layer][kana]
        del p2ss2emb_jp[layer][kana]

## calculate embedding distance
ranking_embed_en_jp = create_embed_ranking(all_combo_en_jp,
                                        p2ss2emb_en,
                                        p2ss2emb_jp,
                                        sr=True, layer=3)
ranking_embed_en_jp.sort(key = lambda x:x[1])
ranking_en_jp.sort(key = lambda x:x[1])
# pearson correlation between ss divergence and embedding distance
pearson_ss_embed(ranking_en_jp, ranking_embed_en_jp)

pearson_layer_cross = []
for i in range(1,13):
    ranking_embed_en_jp = create_embed_ranking(all_combo_en_jp,
                                            p2ss2emb_en,
                                            p2ss2emb_jp,
                                            sr=True, layer=i)
    pearson_layer_cross.append(
        [i, pearson_ss_embed(ranking_en_jp, ranking_embed_en_jp)]
        )

#scratch apd
ranking_en_jp.sort(key = lambda x:x[1])
ranking_embed_en_jp.sort(key = lambda x:x[1], reverse=True)

ss_top15 = [[' | '.join(pair), round(score, 2)] for pair, score in ranking_en_jp[:15]]
embed_top15 = [[' | '.join(pair), round(score, 2)] for pair, score in ranking_embed_en_jp[:15]]

top15 = pd.DataFrame(np.array(ss_top15+embed_top15).reshape(10,6))

top15 = pd.DataFrame([', '.join(ss_top15), ', '.join(embed_top15)],
                     index=['SS-based', "CWE-based"],
                     columns=['Top 15']
                     )
top15.style.to_latex("top15.tex", encoding="UTF-8")

#scratch pad
# visualization

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'sans-serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 9,
})

plt.figure(figsize=(3,1.5))
plt.bar([pair[0]-0.3 for pair in pearson_layer_en],
        [np.abs(pair[1][0]) for pair in pearson_layer_en],
        width=0.3, color = 'darkblue',
        label='en')
plt.bar([pair[0] for pair in pearson_layer_jp],
        [np.abs(pair[1][0]) for pair in pearson_layer_jp],
        width=0.3, color = 'forestgreen',
        label='jp')
plt.bar([pair[0]+0.3 for pair in pearson_layer_cross],
        [np.abs(pair[1][0]) for pair in pearson_layer_cross],
        width=0.3, color = 'gold',
        label='en-jp')
plt.xticks([pair[0] for pair in pearson_layer_en])
plt.yticks([float(item)/100 for item in range(0,45,10)])
plt.ylabel("|Pearson's r|")
plt.xlabel("Layer")
plt.legend(loc='upper left', fontsize='xx-small')
plt.savefig("Pearson.pgf",format='pgf', bbox_inches='tight')



# TODO: process EFCAMDAT and find error cases
"""
efcamdat_fp = os.path.join("..", "..", "Downloads", "efcamdat",
                           "EFCAMDAT_Database.xml")
efcamdat_nobr_fp = os.path.join("..", "..", "Downloads", "efcamdat",
                           "EFCAMDAT_Database_nobr.xml")
# fix unclosed <br> tag by ignoring all of them
with open(efcamdat_fp) as f:
    text = f.read()
    text = re.sub(r"<br>|</br>|<code>|</code>", "", text)
with open(efcamdat_nobr_fp, 'w') as f:
    f.write(text)
tree = ET.parse(efcamdat_nobr_fp)
root = tree.getroot()
essays = root.find('writings')
jpn = essays.findall(".//*[@nationality='jp']/..")
corrections = essays.findall(".//*[@nationality='jp']/..//*[symbol='PR']/..")
corrections[0].attrib['correct']

for cor in corrections:
    print("="*10)
    print("correction from:")
    if cor.find('selection') is not None:
        print(cor.find('selection').text)
    print("to:")
    if cor.find('tag').find('correct') is not None:
        print(cor.find('tag').find('correct').text)

corrections[3].find('tag').find('correct').text
corrections[3].find('selection').find('correct').text

print(corrections[1])
for essay in jpn:
    essay.find(".//[symbol='PR']/../..").text
prep_errors = jpn.findall(".//symbol")
#jpn = essays.findall("./writing/learner[@nationality='jp']")
jpn[8].find(".//*[symbol='PR']/../..").text
"""