# Jigsaw Multilingual Toxic Comment Classification [link](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/overview).

The target of this competition is to build machine learning models that can identify toxicity in online conversations. Used language was 
**{'en': 'english', 'it': 'italian', 'fr': 'french', 'es': 'spanish', 'tr': 'turkish', 'ru': 'russian', 'pt': 'portuguese'}**. The challenging part of the competition is the data are heavily unbalanced, train data has only english text, validation data has only text on 3 languages while test has 6 languages.


# Summary of my approach
```
- Translated Data
- Preprocessing(Not help)
- Pseudo Labelling (not helped me somehow)
- Label smoothing
- Learning Rate scheduling
- Multi-Stage Training (after training on english and translated data, finetuned further on validation dataset for 1-2 epochs.)
- Used models RoBERTa XLM pre-trained
- Also combination of three model on top XLM RoBERTA large
- Finally used Ensemble techinique
```

# What I have learnt 

- [RoBERTa XLM MLM](https://www.kaggle.com/riblidezso/train-from-mlm-finetuned-xlm-roberta-large) based training which was pretrained on domain specific data.
- Best training policy
- Pseudo Labelling techinique

:v:

