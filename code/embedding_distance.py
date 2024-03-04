from transformers import BertTokenizerFast, BertModel
import numpy as np
import os
from tqdm import tqdm
import json

EXCEPTIONS = ['my', 'our', 'her', 'his', 'their', 'your',
              'according to', 'According to', "'s",
              'で', 'でも', 'が', 'だけ', 'ずつ', 'まで', 'ぐらい', 'ばかり']
DATA_DIR = os.path.join('..', 'data')
RESOURCES_DIR = os.path.join(DATA_DIR, 'resources')
OUTPUT_DIR = os.path.join(DATA_DIR, 'embeddings')

"""utility functions"""

capitalize_ss_fp = os.path.join(RESOURCES_DIR, "supersenses_capitalization.tab")
with open(capitalize_ss_fp) as f:
    capitalize_ss = {lower:cap for lower, cap in [item.strip().split('\t') for item in f.readlines()]}

def get_target_list(fp, lang, ensure=True):
    def _add_p(dct, p, sr, f):
        if p not in dct:
            dct[p] = {'sr': {ss: 0 for ss in sorted(list(capitalize_ss.values()))},
                      'f': {ss: 0 for ss in sorted(list(capitalize_ss.values()))}}
        dct[p]['sr'][sr] += 1
        dct[p]['f'][f] += 1
        return dct

    lemma_id = 1 if lang == 'jpn' else 2
    p_dict = dict()
    p = None
    target_list = []
    toks = []
    with open(fp) as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                if line.startswith("# text"):
                    sent = line.split('=')[1].strip().split()
                continue
            line = line.split("\t")
            tok_id = int(line[0]) - 1
            lexcat = line[-1]
            lemma = line[lemma_id]  # 1 for JPN 2 for ENG
            BIO = lexcat.split('-')[0]

            if lexcat == "_" \
                    or (line[13].split('.')[0] != 'p' and BIO != "I_"):
                if type(p) == list:  # MWE ended last line
                    p_dict = _add_p(p_dict, ' '.join(p), sr, f)
                    target_list.append([sent, ' '.join(p), toks, sr, f])
                p = None
                toks = []
                continue

            if BIO == "B":  # MWE begins
                if type(p) == list:  # MWE ended last line
                    p_dict = _add_p(p_dict, ' '.join(p), sr, f)
                    target_list.append([sent, ' '.join(p), toks, sr, f])
                sr = line[13].split('.')[1]
                f = line[14].split('.')[1]
                p = [lemma]
                toks = [tok_id]

            elif BIO == "I_" and type(p) == list:  # inside MWE
                p.append(lemma)
                toks.append(tok_id)

            elif BIO in ["O", "I~"]:  # single word expression or MWE but of different kind (belong *to*)
                if type(p) == list:  # MWE ended last line
                    p_dict = _add_p(p_dict, ' '.join(p), sr, f)
                    target_list.append([sent, ' '.join(p), toks, sr, f])
                sr = line[13].split('.')[1]
                f = line[14].split('.')[1]
                p = lemma
                toks = [tok_id]
                p_dict = _add_p(p_dict, p, sr, f)
                target_list.append([sent, p, toks, sr, f])
    if ensure:
        for sent, p, ids, _, _ in target_list:  # check to see
            as_is = ' '.join([sent[tok] for tok in ids]).lower()
            lemma = p
            if as_is not in EXCEPTIONS:
                assert as_is == lemma

    return p_dict, target_list

def get_cwe(toks, ids, model, tokenizer, ensure=True):
    """given pre-tokenized sentence (=toks),
    get the CWE of the word of interest, whose index is idx.
    Returns list of 12 CWEs, 1 from each layer.

    ids should be in a simple list in 0-index; e.g., [0] or [1,2,3]
    """
    CWEs = []
    target = ' '.join(toks[ids[0]:ids[-1] + 1]).lower()
    encoded = tokenizer.encode_plus(toks,
                                    return_tensors='pt',
                                    is_split_into_words=True)
    wp_ids = []
    for i in ids:
        wp_ids.extend(np.where(np.array(encoded.word_ids()) == i)[0].tolist())
    tok = tokenizer.decode(encoded['input_ids'].squeeze()[wp_ids]).lower()
    if target not in EXCEPTIONS:
        if ensure:
            if tok != target:
                print(tok, target)
            assert tok == target  # making sure we are looking at the correct token
    output = model(encoded['input_ids'],
                   output_hidden_states=True)
    for i in range(1, 13):  # layers 1 to 12
        CWE = output.hidden_states[i].squeeze()[wp_ids].mean(dim=0)
        CWEs.append(CWE)
    return CWEs

def feed_LM(fp, bmodel, btokenizer, lang, ensure):
    p_dict, target_list = get_target_list(fp, lang, ensure)
    p2ss2emb = {i: {p: {'sr': dict(),
                        'f': dict()}
                    for p in p_dict.keys()
                    }
                for i in range(1, 13)
                }

    for sent, p, ids, sr, f in tqdm(target_list):
        CWEs = get_cwe(sent, ids, bmodel, btokenizer)
        CWEs = [cwe.tolist() for cwe in CWEs]
        for i in range(1, 13):
            if sr not in p2ss2emb[i][p]['sr']:
                p2ss2emb[i][p]['sr'][sr] = []
            if f not in p2ss2emb[i][p]['f']:
                p2ss2emb[i][p]['f'][f] = []
            p2ss2emb[i][p]['sr'][sr].append(CWEs[i - 1])
            p2ss2emb[i][p]['f'][f].append(CWEs[i - 1])

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    output_fp = os.path.join(OUTPUT_DIR, f'p2ss2emb_{lang}_mbert.json')
    print('\n'+'='*100+'\n'+f'Saving the results to {output_fp}')
    with open(output_fp, 'w') as f:
        json.dump(p2ss2emb, f)

def main():
    """ENGLISH"""
    print('\n'+'='*100+'\n'+'Loading English Little Prince Corpus...')
    fp = os.path.join(DATA_DIR, 'prince_en_without_1_4_5.conllulex')
    if not os.path.exists(fp):
        raise IOError('\n'+'English Little Prince not found. Please download from:'+'\n'+\
                      'https://github.com/nert-nlp/English-Little-Prince-SNACS/blob/master/prince_en_without_1_4_5.conllulex'
                      )
    model_name = 'bert-base-multilingual-uncased'
    tokenizer_name = 'bert-base-multilingual-uncased'
    model = BertModel.from_pretrained(model_name)
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    print('\n'+'='*100+'\n'+f'Getting CWEs for all prepositions using {model_name}...')
    feed_LM(fp, model, tokenizer, 'eng', True)

    """JAPANESE"""
    print('\n'+'='*100+'\n'+'Loading Japanese Little Prince Corpus...')
    fp = os.path.join(DATA_DIR, 'lpp_jp.conllulex')
    if not os.path.exists(fp):
        raise IOError('\n'+'Japanese Little Prince not found. Please download from:'+'\n'+\
                      'https://github.com/nert-nlp/English-Little-Prince-SNACS/blob/master/prince_en_without_1_4_5.conllulex'
                      )
    print('\n'+'='*100+'\n'+f'Getting CWEs for all prepositions using {model_name}...')
    feed_LM(fp, model, tokenizer, 'jpn', True)

if __name__ == '__main__':
    main()
# etokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
# emodel = BertModel.from_pretrained('bert-base-uncased')
# etokenizer = BertTokenizerFast.from_pretrained('bert-large-uncased')
# emodel = BertModel.from_pretrained('bert-large-uncased')

# jtokenizer = BertTokenizerFast.from_pretrained('cl-tohoku/bert-base-japanese')
# jmodel = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
# jtokenizer = BertTokenizerFast.from_pretrained('cl-tohoku/bert-base-japanese-char-v3')
# jmodel = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-char-v3')
# jtokenizer = BertTokenizerFast.from_pretrained('cl-tohoku/bert-large-japanese')
# jmodel = BertModel.from_pretrained('cl-tohoku/bert-large-japanese')
# jtokenizer = BertTokenizerFast.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# jmodel = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')