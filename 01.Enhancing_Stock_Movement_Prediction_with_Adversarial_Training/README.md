## Adv-ALSTM

The implementation is from the paper
```
@article{feng2019enhancing,
  title={Enhancing Stock Movement Prediction with Adversarial Training},
  author={Feng, Fuli and Chen, Huimin and He, Xiangnan and Ding, Ji and Sun, Maosong and Chua, Tat-Seng},
  journal={IJCAI},
  year={2019}
}
```

### Changes from the origin paper
- Run the experiment code with Korean stock market data not Nasdaq data.
- Not using hinge loss, the model compiled with binary cross entropy loss.
- Dataset provide also regression problem.
- Regularizer coefficient value moves to nearly zero, e.g. 1e-4, 1e-5, 1e-6, 0.
- Modify Learning rate 0.01 to 0.001.
- Split train dataset, validate dataset randomly, test dataset is exclusively split from train dataset, and validate testset

### How to run the code
```bash
# TRAIN
$ python train.py --epochs=150 --use_adversarial=True --verbose=True --is_regression=False

# TEST
$ python test.py --is_regression=False

# tuning hyperparameter
$ python hyperoptimize.py
```

### Experiment result
1. search hyperparameters of ALSTM on the test set to find the best number of hidden state, lags, regularizer. The performance of each model setting is reported on the test set over five independent runs.
  - hidden nums: 16, lags: 5, regularizer: 0
2. search hyperparameters of Adv-ALSTM to find the best beta, and epsilon which is related to adversarial training. This step is also reported over five different runs.
  - beta: 1.0, epsilon: 0.001
  - Accuracy: 0.5334


