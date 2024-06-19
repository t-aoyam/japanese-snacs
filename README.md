# Repository for J-SNACS

This is a repo for the code and data used in our paper "J-SNACS: Adposition and Case Supersenses for Japanese Joshi" to be presented at LREC-COLING 2024.
`./data/` contains Japanese Little Prince (星の王子さま) Corpus manually annotated for adpositions and case markers, following the SNACS guideline.
`./code/` contains the python scripts that replicate the experimental results in the paper.

## TL;DR

If you just need a clean, adjudicated dataset for J-SNACS, please go grab it from `data/lpp_jp.conllulex`!

## Directory structure
    data
    ├── raw
    │   └── chapters0_3.xlsx
    │   └── ...    
    ├── cleaned
    │   ├── chapters_0_3_cleaned.xlsx
    │   └── ...
    ├── lpp_jp.conllulex
    code
    ├── extract_LPP_html.py
    ├── embedding_distance.py
    ├── xlsx2conllulex.py
    └── ss_distribution.py

## Environment
Please use `pip install -r requirements.txt` to install the required packages. `Python 3.7` was used to run all the codes in this repo.
```console
japanese-snacs$ conda create -n jsnacs python=3.7
japanese-snacs$ conda activate jsnacs
(jsnacs) japanese-snacs$ pip install -r requirements.txt
```

## Obtaining the results from the paper
The cleaned-up version of J-SNACS data is already available under `data/lpp_jp.conllulex`; however, to run the codes used in the paper, you will need to download the English Little Prince Corpus from https://github.com/nert-nlp/English-Little-Prince-SNACS/blob/master/prince_en_without_1_4_5.conllulex.
To replicate the results from our paper, particularly Figure 3 and Table 4, please first obtain the relevant CWEs by running the following command.
```console
(jsnacs) japanese-snacs/code$ python embedding_distance.py
```
`.json` files should be saved under `data/embeddings/` directory. Then run the following command to obtain `Figure3.png` and `Table4.txt`.
```console
(jsnacs) japanese-snacs/code$ python ss_distribution.py
```
If everything runs successfully, the resulting figure should look like this:

![Figure3](https://github.com/t-aoyam/japanese-snacs/assets/57016337/d7db5ac2-c626-43cf-841e-0ef5a3b9450b)

## Annotating the data and converting the annotated `.xlsx` files to `.conllulex`
The raw version (without any annotation) of the data can be obtained by scraping the source HTML data.
```console
(jsnacs) japanese-snacs/code$ python extract_lpp_html.py
```
The resulting `.xlsx` files are designed to make the annotation process easier - once you annotate (and preferably adjudicate) the data, you can place the annotated data under `data/cleaned/` (see files under `data/cleaned/` to make sure your files are exactly in the same format).
Finally, you can obtain the `.conllulex` by running the following command.
```console
(jsnacs) japanese-snacs/code$ python xlsx2conllulex.py
```

## Citation
If you use the data and/or the codes in this repo, please remember to cite our paper!
```
@inproceedings{aoyama-etal-2024-j-snacs,
    title = "{J}-{SNACS}: Adposition and Case Supersenses for {J}apanese Joshi",
    author = "Aoyama, Tatsuya  and
      Taguchi, Chihiro  and
      Schneider, Nathan",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.839",
    pages = "9604--9614",
    abstract = "Many languages use adpositions (prepositions or postpositions) to mark a variety of semantic relations, with different languages exhibiting both commonalities and idiosyncrasies in the relations grouped under the same lexeme. We present the first Japanese extension of the SNACS framework (Schneider et al., 2018), which has served as the basis for annotating adpositions in corpora from several languages. After establishing which of the set of particles (joshi) in Japanese qualify as case markers and adpositions as defined in SNACS, we annotate 10 chapters ({\mbox{$\approx$}}10k tokens) of the Japanese translation of Le Petit Prince (The Little Prince), achieving high inter-annotator agreement. We find that, while a majority of the particles and their uses are captured by the existing and extended SNACS annotation guidelines from the previous work, some unique cases were observed. We also conduct experiments investigating the cross-lingual similarity of adposition and case marker supersenses, showing that the language-agnostic SNACS framework captures similarities not clearly observed in multilingual embedding space.",
}
```

## Licensing
The data were obtained from https://www.aozora.gr.jp/cards/001265/files/46817_24670.html, and are free to be shared and redistributed under CC-BY-JP as described in http://creativecommons.org/licenses/by/2.1/jp/.
