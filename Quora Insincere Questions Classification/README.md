# Quora Insincere Questions Classification [link](https://www.kaggle.com/c/quora-insincere-questions-classification/overview).

The objective is to predict whether a question asked on Quora is sincere or not.Some characteristics that can signify that a question is insincere:

* has a non-neutral tone
* is disparaging or inflammatory
* isn't grounded in reality
* uses sexual content

Submissions are evaluated on F1 score between the predicted and the observed targets

# Model architecture 
<p align="center">
 <img src="https://github.com/SumonKantiDey/Kaggle-Competitions/blob/master/Quora%20Insincere%20Questions%20Classification/img/model.png" >
</p>

I have tried to finetune a roBERTa base transformer model to predict whether a Quora question is sincere or insincere.
<p align="center">
 <img src="https://github.com/SumonKantiDey/Kaggle-Competitions/blob/master/Quora%20Insincere%20Questions%20Classification/img/one.png" width="470" >
<img src="https://github.com/SumonKantiDey/Kaggle-Competitions/blob/master/Quora%20Insincere%20Questions%20Classification/img/two.png" width="470" >
</p>
