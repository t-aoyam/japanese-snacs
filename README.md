# Repository for J-SNACS

This is a repo that contains a dataset of Japanese Little Prince Corpus (星の王子さま) manually annotated for adpositions and case markers, following the SNACS guideline.
## TL;DR

If you just need a clean, adjudicated dataset for J-SNACS, please go grab it from `data/lpp_jp.conllulex`!

## Directory structure
    data
    ├── raw
    │   └── ...
    ├── cleaned
    │   ├── data_split.rels
    │   └── filename.conllu
    ├── lpp_jp.conllulex
    code
    ├── extract_LPP_html.py
    ├── embedding_distance.py
    ├── ss_distribution.py
    └── output

## Environment
Please use `pip install -r requirements.txt` to create an environment. `Python 3.7` was used to run all the codes in this repo.

## Obtaining the data
The cleaned-up version of J-SNACS data is already available under `data/lpp_jp.conllulex`; however, to run the codes used in the paper, you will need to download the English Little Prince Corpus from https://github.com/nert-nlp/English-Little-Prince-SNACS/blob/master/prince_en_without_1_4_5.conllulex.
If you would like the raw version (without any annotation) of the data, you can replicate the HTML scraping process by running
```sh
(jsnacs) foo@bar:./japanese-snacs/code$ python extract_lpp_html.py
```
To replicate the results from our paper, particularly Figure 3 and Table 4, please first obtain the relevant CWEs by running the following command.
```console
(jsnacs) foo@bar:./japanese-snacs/code$ python embedding_distance.py
```
`.json` files should be saved under `data/embeddings/` directory. Then run the following command to obtain `Figure3.png` and `Table4.txt`.
```console
(jsnacs) foo@bar:./japanese-snacs/code$ python ss_distribution.py
```
![Figure3](https://github.com/t-aoyam/japanese-snacs/assets/57016337/d7db5ac2-c626-43cf-841e-0ef5a3b9450b)
