# Transformers: Attention Is All You Need

The Transformer architecture, introduced by Vaswani et al. in 2017 in the paper "Attention Is All You Need", revolutionized sequence modeling by dispensing with recurrence and convolutions entirely. Instead, it relies solely on attention mechanisms.

## Core Components

### Self-Attention
Self-attention allows the model to weigh the importance of different tokens in the input sequence when producing a representation for a given token. The scaled dot-product attention is computed as:

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

where Q, K, V are queries, keys, and values derived from linear projections of the input.

### Multi-Head Attention
Rather than performing a single attention function, the model linearly projects the queries, keys and values multiple times with different learned projections. The outputs are concatenated and projected again. This allows the model to jointly attend to information from different representation subspaces.

### Positional Encoding
Because the model contains no recurrence or convolution, positional information must be injected. The original paper used fixed sinusoidal encodings:

PE_(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

Later work explored learned positional embeddings and relative position biases (e.g., T5, Transformer-XL).

## Encoder-Decoder Structure
The original Transformer is an encoder-decoder model. The encoder maps an input sequence to a continuous representation. The decoder generates an output sequence autoregressively, attending to both previous decoder outputs and the encoder outputs via cross-attention.

## Impact
Transformers became the foundation for BERT (encoder-only), GPT (decoder-only), T5 (encoder-decoder), and virtually all modern large language models. Scaling laws later showed that performance improves predictably with model size, data, and compute.
