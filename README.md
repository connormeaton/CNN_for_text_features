# CNN_for_text_features
This repo is a Keras implementation of a CNN for feature generation from text roughly based on [Kim (2014)][1]. The goal of this work is to transform new data into the proper format to be fed into the DialogueGCN model from this [repo][2].

### Model overview:
The model works as such:
- Convert text data to int tokens
- Pass tokens into word2vec embedding
- Dropout
- Extract features from sequences using convolutional layers (3, filter sizes 2, 4, and 5)
- 1D Maxpool
- Flatten
- Concat tensors from each convolution
- Dropout
- Output tensor

### Updates:
- 4/02/2020: I have the model working on SPAFF data, however some questions remain:
	- How to best tokenize data? Should the word:int mapping be conversation-wide or corpus-wide?        - Currently, the output tensor is shape `[5290, 50]`. This is problematic, as I believe the DialogueGCN model is built to except `[n, n, 100]`. Work in progress here...


[1]: https://arxiv.org/pdf/1408.5882.pdf
[2]: https://github.com/cmeaton/DialogueGCN
