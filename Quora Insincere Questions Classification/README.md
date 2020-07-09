# Quora Insincere Questions Classification [link](https://www.kaggle.com/c/quora-insincere-questions-classification/overview).

The objective is to predict whether a question asked on Quora is sincere or not.Some characteristics that can signify that a question is insincere:

* has a non-neutral tone
* is disparaging or inflammatory
* isn't grounded in reality
* uses sexual content

Submissions are evaluated on F1 score between the predicted and the observed targets

I have tried to finetune a roBERTa base transformer model to predict whether a Quora question is sincere or insincere.
### Database architecture
<p align="center">
<img src="https://user-images.githubusercontent.com/16388826/74765962-abe63780-52ae-11ea-9d06-0a18177d58ea.png">
</p>
