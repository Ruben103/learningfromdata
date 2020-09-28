##Assignment 3 - Support Vector Machines

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
The other experiments can be run with the following commands:
<br/> experimentDefaultSettings: ```python3 LFDassignment3_SVM_yourgroup.py <trainset> <testset> 1```
<br/> experimentCParamterCrossValidation: ```python3 LFDassignment3_SVM_yourgroup.py <trainset> <testset> 2```
<br/> experimentCombinatorialCrossValidation: ```python3 LFDassignment3_SVM_yourgroup.py <trainset> <testset> 3```
<br/> experimentLinearKernel: ```python3 LFDassignment3_SVM_yourgroup.py <trainset> <testset> 4```
<br/> experimentFeatures: ```python3 LFDassignment3_SVM_yourgroup.py <trainset> <testset> 4```