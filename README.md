# Instrument-Differentiation-Autoencoding

Finished.

Music autoencoder which uses an unsupervised approach to attempt separating audio into channels by instrument.

Keyword: "attempt."  The idea was nice - since instruments have distinct timbre, I thought that separable convolutional layers could separate them within the same signal, but in practice this is too computationally expensive and ineffective even with weight sharing across the encoder and decoder.  A possible improvement is to punish the model for making overly complex output, but the current results are so unimpressive that this is probably not something that I will put effort towards.  I did get to learn to use PyTorch, though.

Packages used: librosa, PyTorch, TensorFlow (in old version)
