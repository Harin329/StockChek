#### Stonkformers ####

An exploration of the [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani, 2017) paper. Both the Tensorflow and PyTorch Transformer implementation were tested, with a final adaptation for time series modeling.

### Run Docker Container

docker run -it --gpus=all --rm -v $(realpath ~/Desktop/Transformers):/tf/notebooks -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter

### Contents

Attention Is All You Need - Jupyter Notebook
- Tutorial Implementation of Portugese - English Translator in Tensorflow

Stonkformers - Jupyter Notebook
- Applied Implementation of Time Series Stock Price Prediction in PyTorch
- Alpha Vantage Stock Data
- Blue True, Red Prediction