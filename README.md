# Music Generation Using Deep Learning

Multi-layer recurrent neural networks for training and sampling from texts, inspired by [karpathy/char-rnn](https://github.com/karpathy/char-rnn).

### Requirements
Tensorflow, numpy, python3. If it does not work with all this packages then install additional packages using pip command.

You can install pipcommand using
```
$ sudo apt-get intstall python3-pip
```
This code is written in Python 2, and it requires the [Keras](https://keras.io) deep learning library.

### Usage

All input data should be placed in the `data/` directory. The example `input.txt` is taken from the [Nottingham Dataset (Cleaned)](https://github.com/jukedeck/nottingham-dataset).

To train the model with default settings:
```bash
$ python train.py
```

To sample the model:
```bash
$ python sample.py 100
```

Training loss/accuracy is stored in `logs/training_log.csv`.