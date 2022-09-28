import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

df = pd.read_excel("Ch0_3_0506.xlsx")

type2ss = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
freq = defaultdict(lambda: 0)
sr_f_list = []
sr_list = []
f_list = []
num_tok = 0
for i in range(len(df))[:2420]:
    if type(df.loc[i, "Token"]) == str:
        num_tok += 1    
    if type(df.loc[i, "Token"]) == str and type(df.loc[i, "SceneRole_TA"]) == str:
#        tok = df.loc[i, "Token"] + ' (' + df.loc[i, "Romanized"] + ')'
        tok = df.loc[i, "Romanized"]
        sr = df.loc[i, "SceneRole_TA"]
        f = df.loc[i, "Function_TA"]
        if type(sr) == str and type(f) == str:
            sr_f = "_".join([sr, f])
            sr_f_list.append(sr_f)
            sr_list.append(sr)
            f_list.append(f)
            type2ss[tok]['sr'][sr] += 1
            type2ss[tok]['f'][f] += 1
            type2ss[tok]['sr_f'][sr_f] += 1
            freq[tok] += 1


label = list(freq.keys())
values = list(freq.values())
pairs = []
for pair in zip(label, values):
    pairs.append(pair)
pairs.sort(key=lambda x:x[1], reverse=True)

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size': 10,
})

#matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.figure(figsize=(3.3,2.5))
plt.bar([pair[0] for pair in pairs], [pair[1] for pair in pairs])
plt.xticks(rotation=60)
plt.ylabel('count')

plt.savefig("Type_Frequency.pgf",format='pgf', bbox_inches='tight')


len(list(set(df["SceneRole_TA"][:-5]))[:-1])
len(list(set(df["Function_TA"][:-5]))[:-1])
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
plt.figure(figsize=(3.3,2.5))
plt.bar([pair[0] for pair in sr_f_type], [pair[1] for pair in sr_f_type])
plt.xticks(rotation=60)
plt.ylabel('\# construal pairs')
plt.savefig("Construal_Type.pgf",format='pgf', bbox_inches='tight')
