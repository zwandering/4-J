# README

This file introduce the usage and outline of code.

The project code has the following directory structure
~~~
code
│  ADDA.py  # implementation of ADDA
│  Adversarial.py    
│  DANN.py    # implementation of DANN
│  Dataset.py    # implementation of Dataset
│  draw.py    # draw plot function
│  implementation_PR_PL.py    # implementation of PR-PL
│  main.py    
│  models.py    # implementation of models
│  model_PR_PL.py    # implementation of PR-PL model
│  parser.py    
│  README.md    # this file
│  ResNet.py    # implementation of ResNet model
│  search.py    # implementation of Bayes Search
│  SVM.py    # implementation of SVM
│  tca.py    # implementation of TCA
│
├─SEED-IV    # Train Data
~~~

## How to run

To run the baseline and our model, go to path /code, run command:
~~~bash
python main.py --model XXX 
~~~
the model arg have the following selection:

- Conventional ML model:
  - SVM
- Conventional DL models:
  - MLP
  - resnet
- Domain generalization models:
  - IRM
- Domain adaptation models:
  - tca 
  - DANN
  - ADDA
  - prpl

You can also use args like `--lr` or `--batch_size` to tune the hyperparameters.

## Experiment Setup

All experiments are done on **i9-10900X CPU @ 3.70GHz** along with a **GeForce RTX 3090 GPU**.