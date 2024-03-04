import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import os
import spacy
import pykakasi
kks = pykakasi.kakasi()

capitalize_ss_fp = os.path.join("data", "supersenses_capitalization.tab")
with open(capitalize_ss_fp) as f:
    capitalize_ss = {lower:cap for lower, cap in [item.strip().split('\t') for item in f.readlines()]}

data_fp1 = os.path.join("data", "cleaned", "chapters_0_3_cleaned.xlsx")
data_fp2 = os.path.join("data", "cleaned", "chapters_4_6_cleaned.xlsx")
data_fp3 = os.path.join("data", "cleaned", "chapters_7_9_cleaned.xlsx")
data_fp4 = os.path.join("data", "cleaned", "chapters_10_cleaned.xlsx")
df1 = pd.read_excel(data_fp1)
df2 = pd.read_excel(data_fp2)
df3 = pd.read_excel(data_fp3)
df4 = pd.read_excel(data_fp4)

df = pd.concat([df1, df2, df3, df4], ignore_index=True)


type2ss = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
freq = defaultdict(lambda: 0)
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
            type2ss[rom]['sr'][sr] += 1
            type2ss[rom]['f'][f] += 1
            type2ss[rom]['sr_f'][sr_f] += 1
            freq[rom] += 1

label = list(freq.keys())
values = list(freq.values())
pairs = []
for pair in zip(label, values):
    pairs.append(pair)
pairs.sort(key=lambda x:x[1], reverse=True)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'sans-serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 9,
})

#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.figure(figsize=(3,1.5))
plt.bar([pair[0] for pair in pairs[:15]], [pair[1] for pair in pairs[:15]])
plt.xticks(rotation=75)#fontdict={'family': 'sans-serif', 'size': 9})
plt.ylabel('Frequency')

plt.savefig("Type_Frequency.pgf",format='pgf', bbox_inches='tight')


len(list(set(df["SR"]))[:-1])
len(list(set(df["F"]))[:-1])
len(set(sr_f_list))
same = 0
for item in set(sr_f_list):
    sr, f = item.split('_')
    if sr == f:
        same += 1
len(sr_f_list)

sr_f_type = []
for key in type2ss:
    type_num = len(list(type2ss[key]['sr_f'].keys()))
    sr_f_type.append((key, type_num))

sr_f_type.sort(key=lambda x:x[1], reverse=True)
plt.figure(figsize=(3,1.5))
plt.bar([pair[0] for pair in sr_f_type[:15]], [pair[1] for pair in sr_f_type[:15]])
plt.xticks(rotation=75)
plt.ylabel('\# Construal Pairs')
plt.savefig("Construal_Type.pgf",format='pgf', bbox_inches='tight')
