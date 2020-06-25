# Jigsaw Multilingual Toxic Comment Classification [link](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/overview).

The target of this competition is to build machine learning models that can identify toxicity in online conversations. Used language in this competition dataset was 
**{'en': 'english', 'it': 'italian', 'fr': 'french', 'es': 'spanish', 'tr': 'turkish', 'ru': 'russian', 'pt': 'portuguese'}**. The challenging part of the competition is the data are heavily unbalanced, train data has only english text, validation data has only text on 3 languages while test has 6 languages.


# Summary of my approach
```
- Used Translated Data
- Preprocessing (Not helped)
- Pseudo labeling
- Label smoothing
- Learning Rate scheduling
- Multi-Stage Training (after training on english and translated data, finetuned further on validation dataset for 1-2 epochs.)
- Used  RoBERTa XLM pre-trained models
- Different architectures on top of XLM-Roberta
- Finally used Ensemble Gmean of top score public submissions
```

# What I have learnt 

- [RoBERTa XLM MLM](https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large) based training which was pretrained on domain specific data.
- Best training policy
- Pseudo Labeling techinique

:v: I have achieved Bronze model in this competition and the position was **(151/1621)**.

