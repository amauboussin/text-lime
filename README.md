# Explaining the predictions of any (text) classifier

An extension of LIME [Ribeiro et. al 2016](https://arxiv.org/abs/1602.04938) for text classification.
The main pipeline trains words embeddings using [fastText](https://github.com/facebookresearch/fastText), add noises to embedding space of documents,
and identifies important words by analyzing how the noise affects the output probabilities of any text classification model. The results are visualized in d3.js.

