# KCT-STN
KCT-STN(KAN-Convolutional-Transformer SpatioTemporal Network) is a spatiotemporal network for driving fatigue detection with self-supervised learning.  
In this repo you can see the model and a uniform framework for parsing and training the eeg data.

![structure of KCT-STN](images\structure.png)

## Performance Comparison

We compare the performance with different methods in [Dataset Cao et al](https://www.nature.com/articles/s41597-019-0027-4). Below is the result:

| Method                 | Accuracy | Recall | Precision |
|------------------------|----------|--------|-----------|
| EEGNet             | 82.52%   | 83.96% | 83.30%    |
| EEGTransformer     | 78.02%   | 72.22% | 78.56%    |
| InterpretableCNN   | 77.70%   | 75.30% | 74.66%    |
| ESTCNN             | 77.79%   | 75.01% | 79.12%    |
| **KCT-STN (ours)**   | **87.66%** | **88.13%** | **87.62%** |

The results show that our proposed **KCT-STN** method outperforms other models in terms of accuracy, recall, and precision.



## Env package version information
python: pyhon 3.7
cuda: 11.3
pytorch: 1.12.1
mne: 1.3.1

## Quick Start
1. git clone https://github.com/iuayjh/KCT-STN.git
2. install the denpendcy.  
ps: The basic denpendcy can refer the env information part
3. run the train files under the training fold.  
ps: This code just for personal study, and I perfer to use the ide to run the code immediately. So I don't use the argparse and write a .sh file to start the code.