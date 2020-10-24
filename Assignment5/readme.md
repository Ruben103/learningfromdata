##Assignment 5 - Hyperparameter tuning for Noun-Noun compound classification and Word Embeddings

###Robin Schut and Sharif Hamed

---
####Setting up a virtual environment
```virtualenv venv```
<br/>```source venv/bin/activate```
<br/>```pip3 install -r requirements.txt```
---
####Run Instructions
```python3 LFDassignment3_SVM_yourgroup.py <trainset> <testset>```

There are several experiments in the code. The experiment that shows our best model can be run with the above command.
The other experiments need to be commented in the file ```LFDassignment5_12.py``` These are the following experiments with self-explanatory name
* experimentDropoutRate
* experimentOptimisers
* experimentBatchsize
