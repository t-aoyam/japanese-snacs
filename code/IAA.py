import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score, accuracy_score

#data_fp = os.path.join("data", "chapters0_3_ADJ.xlsx")
data_fp = os.path.join("data", "chapters4_6_ADJ.xlsx")
ss_fp = os.path.join("data", "supersenses.txt")

# TA as anno1, CT as anno2

data = pd.read_excel(data_fp)
sr_ct_all = data["SceneRole_CT"].to_list()
f_ct_all = data["Function_CT"].to_list()
sr_ta_all = data["SceneRole_TA"].to_list()
f_ta_all = data["Function_TA"].to_list()

# p, r, f for annotation target
target_ct = set([i for i, item in enumerate(sr_ct_all) if not pd.isna(item)])
target_ta = set([i for i, item in enumerate(sr_ta_all) if not pd.isna(item)])
p = len(target_ta.intersection(target_ct))/len(target_ct)
r = len(target_ta.intersection(target_ct))/len(target_ta)
f = 2*p*r/(p+r)

# kappa for SR, F, construal, missing annotation as "?"
with open(ss_fp) as f:
    ss = [item.strip('\n ').lower() for item in f.readlines()]

sr_cts = []
f_cts = []
sr_tas = []
f_tas = []
c = set()
for s1 in ss:
    for s2 in ss:
        c.add(' '.join([s1,s2]))

for i in range(len(sr_ct_all)):
    if pd.isna(sr_ct_all[i]) and pd.isna(sr_ta_all[i]):
        continue
    sr_ct = sr_ct_all[i].lower() if not pd.isna(sr_ct_all[i]) else "?"
    sr_ta = sr_ta_all[i].lower() if not pd.isna(sr_ta_all[i]) else "?"
    f_ct = f_ct_all[i].lower() if not pd.isna(f_ct_all[i]) else "?"
    f_ta = f_ta_all[i].lower() if not pd.isna(f_ta_all[i]) else "?"
    sr_cts.append(sr_ct)
    sr_tas.append(sr_ta)
    f_cts.append(f_ct)
    f_tas.append(f_ta)
c_cts = [' '.join([sr, f]) for sr, f in zip(sr_cts, f_cts)]
c_tas = [' '.join([sr, f]) for sr, f in zip(sr_tas, f_tas)]

accuracy_score(sr_tas, sr_cts)
accuracy_score(f_tas, f_cts)
accuracy_score(c_tas, c_cts)

cohen_kappa_score(sr_tas, sr_cts, labels=ss)
cohen_kappa_score(f_tas, f_cts, labels=ss)
cohen_kappa_score(c_tas, c_cts, labels=list(c))
