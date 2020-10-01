# Quora Insincere Questions Classification [link](https://www.kaggle.com/c/quora-insincere-questions-classification/overview).


[![Alt text](https://img.youtube.com/vi/VID/0.jpg)](https://www.youtube.com/watch?v=VID)

The objective is to predict whether a question asked on Quora is sincere or not. Some characteristics that can signify that a question is insincere:

* has a non-neutral tone.
* is disparaging or inflammatory.
* isn't grounded in reality.
* uses sexual content.

Submissions are evaluated on F1 score between the predicted and the observed targets
## Solution summary
```
Preprocessing: Cleaning special characters, number pre-processing, misspell cleaning.
Embedding: GLoVe, FastText, Paragram, word2vec embeddings 
Neural Network architecture: 
  - 5 folds stacked GRU-60 hidden units with attention and (GLoVe+FastText+Paragram+word2vec) embeddings. Acc: .69233 [attention-based-gru-101.ipynb]
  - 5 folds stacked GRU-128,64 hidden units with attention, Convolutions with Glove+Paragram embeddings. Acc: .66231 [kfold-bi-lstm-bi-gru.ipynb]
```
## Model architecture (5 folds stacked GRU-60 hidden units with attention)
<p align="center">
 <img src="https://github.com/SumonKantiDey/Kaggle-Competitions/blob/master/Quora%20Insincere%20Questions%20Classification/img/model.png" >
</p>

I have played a bit around with finetune [RoBERTa base transformer model](https://github.com/SumonKantiDey/Kaggle-Competitions/blob/master/Quora%20Insincere%20Questions%20Classification/roberta-insincerity-check.ipynb) and my aim is to load the model after training which will take a question 
as input and return question label as sincere or insincere.
<p align="center">
 <img src="https://github.com/SumonKantiDey/Kaggle-Competitions/blob/master/Quora%20Insincere%20Questions%20Classification/img/one.png" width="470" >
<img src="https://github.com/SumonKantiDey/Kaggle-Competitions/blob/master/Quora%20Insincere%20Questions%20Classification/img/two.png" width="470" >
</p>
<p>
[![Demo CountPages alpha](https://share.gifyoutube.com/KzB6Gb.gif)](https://www.youtube.com/watch?v=ek1j272iAmc)
</p>
