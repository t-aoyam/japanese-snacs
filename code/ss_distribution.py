import os
import re
import pykakasi
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import json
from numpy import dot
from numpy.linalg import norm
import torch
from embedding_distance import get_target_list
kks = pykakasi.kakasi()

DATA_DIR = os.path.join('..', 'data')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')

JPN_MWE = {'ni totsu te': 'nitotte', 'to ha': 'toha', 'ni tsui te': 'nitsuite'}
# convert both distributions to relative frequencies
def abs2rel(p_dict):
    p_dict_rel = dict()
    for p in p_dict:
        p_dict_rel[p] = dict()
        p_dict_rel[p]['sr'] = [(sr, freq/sum(p_dict[p]['sr'].values()))\
                             for sr, freq in p_dict[p]['sr'].items()]
        p_dict_rel[p]['f'] = [(f, freq/sum(p_dict[p]['f'].values()))\
                            for f, freq in p_dict[p]['f'].items()]
    return p_dict_rel

# compute JS divergence
def measure_divergence(ss_dist1, ss_dist2):
    values1 = [item[1] for item in ss_dist1]
    keys1 = [item[0] for item in ss_dist1]
    values2 = [item[1] for item in ss_dist2]
    keys2 = [item[0] for item in ss_dist2]
    assert keys1 == keys2  # ensuring that the order is the same
    return jensenshannon(values1, values2)

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

def create_cwe_ranking(all_combo, p2ss2emb1, p2ss2emb2=None,
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


def get_pearson_by_layer(all_combo, ranking_ss, p2ss2emb1, p2ss2emb2=None):
    pearson_layer = []
    for i in range(1, 13):
        ranking_cwe = create_cwe_ranking(all_combo, p2ss2emb1, p2ss2emb2,
                                         sr=True, layer=i)
        pearson_layer.append(
            [i, pearson_ss_embed(ranking_ss, ranking_cwe)]
        )
    return pearson_layer

def kana2roman(p2ss2emb_jpn, kks):
    ps = list(p2ss2emb_jpn['1'].keys())
    kana2roman_dict = {p: ''.join([item['passport'] for item in kks.convert(re.sub(' ', '', p))])
                for p in ps}
    for layer in p2ss2emb_jpn:
        for kana in ps:
            p2ss2emb_jpn[layer][kana2roman_dict[kana]] = p2ss2emb_jpn[layer][kana]
            del p2ss2emb_jpn[layer][kana]
    return p2ss2emb_jpn

def ranking2text(ranking_ss, ranking_cwe, k=15):
    ranking_ss.sort(key = lambda x:x[1])
    ranking_cwe.sort(key = lambda x:x[1], reverse=True)

    ss_top15 = [['SS', ' <-> '.join(pair), str(round(score, 2))] for pair, score in ranking_ss[:k]]
    cwe_top15 = [['CWE', ' <-> '.join(pair), str(round(score, 2))] for pair, score in ranking_cwe[:k]]
    output = ['\t'.join(['Metrics', 'Pair', 'Score'])]
    for ss in ss_top15:
        output.append('\t'.join(ss))
    for cwe in cwe_top15:
        output.append('\t'.join(cwe))
    fp = os.path.join(DATA_DIR, 'Table4.txt')
    text = '\n'.join(output)
    with open(fp, 'w') as f:
        f.write(text)
    print('\n'+'='*100+'\n'+f'Table 4 saved to {fp}!')

def get_figure(pearson_layer_eng,
               pearson_layer_jpn,
               pearson_layer_xling):
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'pgf.rcfonts': False,
        'font.size': 9,
    })

    plt.figure(figsize=(3,1.5))
    plt.bar([pair[0]-0.3 for pair in pearson_layer_eng],
            [np.abs(pair[1][0]) for pair in pearson_layer_eng],
            width=0.3, color = 'darkblue',
            label='en')
    plt.bar([pair[0] for pair in pearson_layer_jpn],
            [np.abs(pair[1][0]) for pair in pearson_layer_jpn],
            width=0.3, color = 'forestgreen',
            label='jp')
    plt.bar([pair[0]+0.3 for pair in pearson_layer_xling],
            [np.abs(pair[1][0]) for pair in pearson_layer_xling],
            width=0.3, color = 'gold',
            label='en-jp')
    plt.xticks([pair[0] for pair in pearson_layer_eng])
    plt.yticks([float(item)/100 for item in range(0,45,10)])
    plt.ylabel("|Pearson's r|")
    plt.xlabel("Layer")
    plt.legend(loc='upper left', fontsize='xx-small')
    fp = os.path.join(DATA_DIR, "Figure3.png")
    plt.savefig(fp, format='png', bbox_inches='tight')
    print('\n'+'='*100+'\n'+f'Figure 3 saved to {fp}!')

def main():

    # ENGLISH

    # load the pre-computed embedding distance
    fp = os.path.join(EMBEDDINGS_DIR, 'p2ss2emb_eng_mbert.json')
    if not os.path.exists(fp):
        raise IOError('\n'+'English pre-computed embeddings not found. Please first run `embedding_distance.py`.')
    print('\n'+'='*100+'\n'+'Loading the pre-computed CWEs...')
    with open(fp) as f:
        p2ss2emb_eng = json.loads(f.read())

    print('\n'+'='*100+'\n'+'Loading English Little Prince Corpus...')
    fp = os.path.join(DATA_DIR, 'prince_en_without_1_4_5.conllulex')
    if not os.path.exists(fp):
        raise IOError('\n'+'English Little Prince not found. Please download from:'+'\n'+\
                      'https://github.com/nert-nlp/English-Little-Prince-SNACS/blob/master/prince_en_without_1_4_5.conllulex'
                      )
    p_eng, _ = get_target_list(fp, 'eng', ensure=True)
    p_eng_rel = abs2rel(p_eng)
    all_combo_eng = create_all_combo(p_eng.keys())
    # compute ss distribution divergence
    print('\n'+'='*100+'\n'+'Computing similarities...')
    ranking_ss_eng = create_ss_ranking(all_combo_eng, p_eng_rel, p_eng_rel, sr=True)
    ranking_cwe_eng = create_cwe_ranking(all_combo_eng, p2ss2emb_eng, sr=True, layer=12)
    # pearson correlation between ss divergence and embedding distance
    # TODO pearson_ss_embed(ranking_eng, ranking_embed_eng)
    pearson_by_layer_eng = get_pearson_by_layer(all_combo_eng, ranking_ss_eng, p2ss2emb_eng)

    # JAPANESE

    # load the pre-computed embedding distance
    print('\n'+'='*100+'\n'+'Loading the pre-computed CWEs...')
    fp = os.path.join(EMBEDDINGS_DIR, 'p2ss2emb_jpn_mbert.json')
    if not os.path.exists(fp):
        raise IOError('\n'+'Japanese pre-computed embeddings not found. Please first run `embedding_distance.py`.')
    with open(fp) as f:
        p2ss2emb_jpn = json.loads(f.read())
    # needs some postprocessing to change from kana to romanized for J
    p2ss2emb_jpn = kana2roman(p2ss2emb_jpn, kks)

    print('\n'+'='*100+'\n'+'Loading Japanese Little Prince Corpus...')
    fp = os.path.join(DATA_DIR, 'lpp_jp.conllulex')
    if not os.path.exists(fp):
        raise IOError('\n'+'Japanese Little Prince not found. Please download from:'+'\n'+\
                      'https://github.com/t-aoyam/japanese-snacs/blob/main/data/lpp_jp.conllulex'
                      )
    p_jpn, _ = get_target_list(fp, 'eng', ensure=False)  # 'eng' because of .conllulex column,
                                                         # no ensuring because kana != roman
    for p in p_jpn:
        if p in JPN_MWE:
            p_jpn[JPN_MWE[p]] = p_jpn[p]
            del p_jpn[p]

    p_jpn_rel = abs2rel(p_jpn)

    all_combo_jpn = create_all_combo(p_jpn.keys())

    # compute ss distribution divergence
    print('\n'+'='*100+'\n'+'Computing similarities...')
    ranking_ss_jpn = create_ss_ranking(all_combo_jpn, p_jpn_rel, p_jpn_rel, sr=True)
    ranking_cwe_jpn = create_cwe_ranking(all_combo_jpn, p2ss2emb_jpn, sr=True, layer=12)
    # pearson correlation between ss divergence and embedding distance
    # TODO pearson_ss_embed(ranking_jpn, ranking_embed_jpn)
    # by layer
    pearson_by_layer_jpn = get_pearson_by_layer(all_combo_jpn, ranking_ss_jpn, p2ss2emb_jpn)

    """CROSS LINGUAL"""
    all_combo_xling = create_all_combo(list(p_eng.keys()), list(p_jpn.keys()))
    # compute ss distribution divergence
    print('\n'+'='*100+'\n'+'Computing cross-lingual similarities...')
    ranking_ss_xling = create_ss_ranking(all_combo_xling,
                                      p_eng_rel, p_jpn_rel, sr=True)

    # calculate embedding distance
    ranking_cwe_xling = create_cwe_ranking(all_combo_xling,
                                            p2ss2emb_eng,
                                            p2ss2emb_jpn,
                                            sr=True, layer=3)  # TODO make sure the layer
    ranking_cwe_xling.sort(key = lambda x:x[1])
    ranking_ss_xling.sort(key = lambda x:x[1])
    # pearson correlation between ss divergence and embedding distance
    pearson_ss_embed(ranking_ss_xling, ranking_cwe_xling)
    pearson_by_layer_xling = get_pearson_by_layer(all_combo_xling, ranking_ss_xling,
                                                  p2ss2emb_eng, p2ss2emb_jpn)
    get_figure(pearson_by_layer_eng, pearson_by_layer_jpn, pearson_by_layer_xling)
    ranking2text(ranking_ss_xling, ranking_cwe_xling)

if __name__ == "__main__":
    main()