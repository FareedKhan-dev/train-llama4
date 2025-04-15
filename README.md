<!-- omit in toc -->
# Building LLaMA 4 from Scratch with Python
LLaMA 4 has already faced criticism, as [some Reddit users](https://www.reddit.com/r/LocalLLaMA/comments/1jsl37d/im_incredibly_disappointed_with_llama4/?rdt=40510) claimed that it couldn’t perform tasks that models already 6 months old can do. Though it is a separate debate, LLaMA 4, in its series, is a new step after Mistral that showcases the strengths of MoE-based models.

In this blog, we are going to create the LLaMA 4 MoE architecture step by step in **jupyter notebook** from scratch to understand how it is actually created.

Following is the output of our trained 2.2 million-parameter LLaMA MoE on a tiny English dataset for 3000 Epochs (Colab T4 GPU).

```
Input: Alice

Output: Alice 'without pictures or conversation?'
So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure
of making a daisy-chain wo ...
```
<!-- omit in toc -->
## Table of Contents

- [Llama 4 MoE Architecture Overview](#llama-4-moe-architecture-overview)
- [Setting Up the Stage](#setting-up-the-stage)
- [Define the Training Corpus](#define-the-training-corpus)
- [Character-Level Tokenization](#character-level-tokenization)
- [Encode the Corpus](#encode-the-corpus)
- [Define Hyperparameters](#define-hyperparameters)
- [Data Preparation for Training](#data-preparation-for-training)
- [Batching Strategy (Random Sampling)](#batching-strategy-random-sampling)
- [Model Component Initialization](#model-component-initialization)
- [Rotary Positional Embedding (RoPE) Precomputation](#rotary-positional-embedding-rope-precomputation)
- [RMSNorm Layers Initialization](#rmsnorm-layers-initialization)
    - [Attention Layers Initialization (MHA)](#attention-layers-initialization-mha)
- [Mixture-of-Experts (MoE) Layers Initialization](#mixture-of-experts-moe-layers-initialization)
- [Final Output Layer Initialization](#final-output-layer-initialization)
- [Causal Mask Precomputation](#causal-mask-precomputation)
- [Training Setup](#training-setup)
- [Define Loss Function](#define-loss-function)
- [Training the Model](#training-the-model)
- [Text Generation](#text-generation)
- [The Generation Loop](#the-generation-loop)
- [Decode Generated Sequence](#decode-generated-sequence)
- [Save Model State (Optional)](#save-model-state-optional)
- [Conclusion](#conclusion)

## Llama 4 MoE Architecture Overview

Let’s first understand the LLaMA 4 architecture as an intermediate techy person, and then use an example **“the cat sat”** to see how it goes through the architecture to get a clear understanding of it.

Imagine you have a really tough job. Instead of hiring one person who *kinda* knows everything, you hire a team of specialists, each amazing at one particular thing (like an electrician, a plumber, a painter). You also hire a manager who looks at the current task and sends it to the right specialist(s).

MoE in AI models is kinda like that. Instead of one gigantic neural network trying to learn everything, an MoE layer has:

1.  **A Team of “Experts”**: These are smaller, specialized neural networks (usually simple Feed-Forward Networks or MLPs). Each expert might get good at handling certain types of information or patterns.
2.  **A “Router” (The Manager)**: This is another small network. Its job is to look at the input data (like a word or part of a word) and decide which expert(s) are the best fit to handle it right now.

![llama 4 architecture High level overview](https://miro.medium.com/v2/resize:fit:1250/1*2mDFOnGkEuE20LVsV162fQ.png)

Imagine our model is processing the sentence: The cat sat.

1.  Tokens: First, we break it into pieces (tokens): “The” “cat” “sat”
2.  Router Gets a Token: The MoE layer receives the token `cat` (represented as a bunch of numbers, an embedding vector). The `Router` looks at this `cat` vector.
3.  Router Chooses: Let's say we have 4 experts (`E1`, `E2`, `E3`, `E4`). The `Router` decides which ones are best suited for `cat`.
4.  Maybe it thinks `E2` (perhaps good with nouns?) and `E4` (perhaps good with animal concepts?) are the top choices. It gives scores or "weights" to these choices (e.g., 70% for `E2`, 30% for `E4`).

![Processing of cat input](https://miro.medium.com/v2/resize:fit:1250/1*7j2VkTYVNeAgxXHMjAAy2A.png)

The `cat` vector is sent only to `Expert 2` and `Expert 4`. `Experts 1` and `3` don't do any work for this token, saving computation! `E2` processes `cat` and generates its result (`Output_E2`). `E4` processes `cat` and generates its result (`Output_E4`).

![Chosen Experts for cat word](https://miro.medium.com/v2/resize:fit:875/1*5aDQsQKmNECI6BWXTUgTjw.png)

We now combine the results from the chosen experts using the `router` weights: `Final_Output = (0.7 * Output_E2) + (0.3 * Output_E4).`

This `Final_Output` is what the `MoE` layer passes on for the token `cat`. This happens for every token in the sequence! Different tokens might get routed to different experts.

So, when our model processes text like `"The cat sat."`, the overall journey looks like this:

![LLaMA 4 detailed architecture](https://miro.medium.com/v2/resize:fit:1250/1*0gJvnms7Tq0QXwskhW2ohA.png)

`Input Text` goes into the `Tokenizer`.`Tokenizer` creates numerical `Token IDs`. `Embedding Layer` turns IDs into meaningful number vectors (`Embeddings`) and adds `Positional Info` (using `RoPE` later in attention).

These vectors go through multiple `Transformer Blocks`. Each block has:

*   `Self-Attention` (where tokens look at each other, enhanced by `RoPE`).
*   `MoE Layer` (where the `router` sends tokens to specific `experts`).
*   `Normalization` (`RMSNorm`) and `Residual connections` help learning.

The output from the last block goes to a `Final Layer`. This layer produces `Logits` (scores) for every possible next token in our vocabulary.

We convert `logits` to `Probabilities` and `Predict the Next Token`.

Now that we have a feel for how `MoE` fits into the picture, let’s dive into the code and build these components step-by-step! We’ll start by setting up our coding environment.

## Setting Up the Stage

Before we start coding the model, we need to import the module we’ll use, so let’s do that first.

```python
# Import necessary libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import math
import os
import collections # For BPE-like processing if extended
import re          # For initial splitting

# --- Device Configuration ---
# Theory: Set the device (GPU 'cuda' if available, else CPU) for tensor operations.
# This ensures models and data are processed efficiently on available hardware.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
print("Libraries imported and device configured.")


### OUTPUT ###
# PyTorch version: 2.6.0+cu124 # (Example output, might vary)
# Using device: cuda
# Libraries imported and device configured.
```

The output confirms we’ve successfully imported the libraries. I will be using a Colab T4 GPU to train the model. If you want to train on a cheaper GPU, reduce the number of epochs.

## Define the Training Corpus

We need some text data to train our language model. A real model like LLaMA 4 is trained on trillions of words!

For our tiny example, just to see how the code works, we’ll use a small paragraph from Lewis Carroll’s “Alice’s Adventures in Wonderland”. This small size lets us easily track what’s happening.

```python
# Define the raw text corpus for training
corpus_raw = """
Alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing to do: once or twice she had peeped into the
book her sister was reading, but it had no pictures or conversations in
it, 'and what is the use of a book,' thought Alice 'without pictures or
conversation?'
So she was considering in her own mind (as well as she could, for the
hot day made her feel very sleepy and stupid), whether the pleasure
of making a daisy-chain would be worth the trouble of getting up and
picking the daisies, when suddenly a White Rabbit with pink eyes ran
close by her.
"""

print(f"Training corpus defined (length: {len(corpus_raw)} characters).")


### OUTPUT ###
# Training corpus defined (length: 593 characters).
```

This simply defines the corpus_raw string variable holding our sample text and prints its total length (593 characters, including spaces, newlines, and punctuation).

## Character-Level Tokenization

Computers don’t understand letters, they understand numbers. Tokenization is the process of converting text into numbers (tokens) that a model can process. We’ll use the simplest method: character-level tokenization.

1.  Find every unique character in our `corpus_raw`.
2.  Assign a unique integer ID to each unique character.
3.  Create mappings (dictionaries) to convert characters to IDs (`char_to_int`) and IDs back to characters (`int_to_char`). The total count of unique characters is our `vocab_size`.

![Tokenization Process](https://miro.medium.com/v2/resize:fit:1250/1*wF1mmcnksfgCms8673U7Ow.png)

```python
# Find all unique characters in the raw corpus
chars = sorted(list(set(corpus_raw)))
vocab_size = len(chars)

# Create character-to-integer mapping (encoding)
char_to_int = { ch:i for i,ch in enumerate(chars) }

# Create integer-to-character mapping (decoding)
int_to_char = { i:ch for i,ch in enumerate(chars) }

print(f"Created character vocabulary of size: {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")
# Optional: Print mappings
# print(f"Char-to-Int mapping sample: {{k: char_to_int[k] for k in list(char_to_int)[:5]}}")
# print(f"Int-to-Char mapping sample: {{k: int_to_char[k] for k in list(int_to_char)[:5]}}")


### OUTPUT ###
# Created character vocabulary of size: 36
# Vocabulary:
#  '(),-.:?ARSWabcdefghiklmnoprstuvwy
```

The code found 36 unique characters (including newline `\n`, space, punctuation, uppercase, and lowercase letters) in our small corpus.

This `vocab_size` is important for setting up our model layers later. It also created the `char_to_int` and `int_to_char` dictionaries for conversion and printed the full list of characters in our vocabulary.

## Encode the Corpus

Now we use the char_to_int mapping we just created to convert the entire corpus_raw string into a sequence of corresponding integer IDs.

This numerical representation is what the model will actually train on. We store this sequence as a PyTorch tensor for efficiency.

```python
# Encode the entire corpus into a list of integer IDs
encoded_corpus = [char_to_int[ch] for ch in corpus_raw]

# Convert the list into a PyTorch tensor
full_data_sequence = torch.tensor(encoded_corpus, dtype=torch.long, device=device)

print(f"Encoded corpus into a tensor of shape: {full_data_sequence.shape}")
# Optional: Display first few encoded IDs
# print(f"First 50 encoded token IDs: {full_data_sequence[:50].tolist()}")


### OUTPUT ###
# Encoded corpus into a tensor of shape: torch.Size([593])
```

Our 593-character text has been successfully converted into a single PyTorch tensor (basically a list of numbers) of length 593. Each number in this tensor represents a character from the original text. It’s also placed on the device we specified earlier (e.g., ‘cuda’).

## Define Hyperparameters

Next w need to define the hyperparameters settings that we choose before training. They define the model’s architecture (how big it is, how many layers, etc.) and how it learns. For our `LLaMA 4`-like model, key hyperparameters include:

*   `d_model`: The main dimension used throughout the model (size of token embeddings and hidden states).
*   `n_layers`: How many Transformer blocks are stacked on top of each other. More layers usually mean a more powerful (but slower) model.
*   `n_heads`: Number of parallel attention calculations (heads) in the Multi-Head Attention mechanism. `d_model` must be divisible by `n_heads`.
*   `block_size`: The maximum length of the input sequence the model looks at during training (also called context length).
*   `rms_norm_eps`: A small value added for numerical stability in `RMSNorm`.
*   `rope_theta`: A parameter controlling the frequencies used in `Rotary Positional Embeddings`.

`MoE` parameters:

*   `num_local_experts`: How many "expert" MLPs are in each `MoE` layer.
*   `num_experts_per_tok`: How many experts the router sends each token to (Top-K routing).
*   `intermediate_size_expert/shared`: The hidden dimension inside the expert/shared MLPs.

We are using much smaller values than the real LLaMA 4 for this demonstration to make it run quickly on typical hardware.

```python
# --- Model Architecture Hyperparameters ---
# vocab_size is already determined from the data
d_model = 128         # Embedding dimension (reduced significantly)
n_layers = 4          # Number of Transformer blocks (reduced)
n_heads = 4           # Number of attention heads
block_size = 64       # Maximum context length (sequence length)
rms_norm_eps = 1e-5   # Epsilon for RMSNorm stability
rope_theta = 10000.0  # Theta parameter for RoPE (reduced from Llama 4's 500k)

# --- MoE Specific Hyperparameters ---
num_local_experts = 4      # Number of experts per MoE layer (reduced from 16)
num_experts_per_tok = 2   # Number of experts to route each token to (Top-K, reduced from 4?)
intermediate_size_expert = d_model * 2  # Hidden dimension within each expert MLP (scaled down)
intermediate_size_shared = d_model * 2  # Hidden dimension within the shared MLP (scaled down)

# --- Attention Hyperparameters ---
# d_k (dimension per head) will be derived from d_model and n_heads

# --- Training Hyperparameters ---
learning_rate = 5e-4  # Learning rate
batch_size = 16       # Number of sequences processed in parallel
epochs = 3000         # Number of training iterations (adjust as needed)
eval_interval = 300  # How often to print loss

# --- Derived Hyperparameters ---
assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
d_k = d_model // n_heads # Dimension of keys/queries/values per head
expert_dim = intermediate_size_expert # Alias for clarity
shared_expert_dim = intermediate_size_shared # Alias for clarity
```

Let’s look at all the parameter values, we have just coded.

```
--- Hyperparameters Defined ---
Vocabulary Size (vocab_size): 36
Embedding Dimension (d_model): 128
Number of Layers (n_layers): 4
Number of Attention Heads (n_heads): 4
Dimension per Head (d_k): 32
Max Sequence Length (block_size): 64
RMSNorm Epsilon (rms_norm_eps): 1e-05
RoPE Theta (rope_theta): 10000.0

--- MoE Specific ---
Number of Local Experts (num_local_experts): 4
Experts per Token (num_experts_per_tok): 2
Expert Intermediate Size (expert_dim): 256
Shared MLP Intermediate Size (shared_expert_dim): 256

--- Training Specific ---
Learning Rate: 0.0005
Batch Size: 16
Epochs: 3000
```

This output clearly lists all the configuration values we just set for our model and training process. We can see the model dimensions (like `d_model=128`), the number of `MoE` experts (`4`), the number of experts each token will use (`2`), the context window (`block_size=64`), and the training parameters (`learning_rate=0.0005`, `batch_size=16`, `epochs=3000`).

## Data Preparation for Training

Language models like ours learn by predicting the next token in a sequence given the tokens that came before it. To prepare the data for this, we slide a window of length `block_size` across our `full_data_sequence`.

1.  The input (`x`) is a chunk of `block_size` tokens.
2.  The target (`y`) is the same chunk shifted one position to the right.
3.  So, for every token in the input `x`, the model's goal is to predict the token at the same position in the target `y`.

We create all possible overlapping chunks like this from our corpus.

```python
# Create lists to hold all possible input (x) and target (y) sequences
all_x = []
all_y = []

# Iterate through the encoded corpus tensor to extract overlapping sequences
num_total_tokens = len(full_data_sequence)
for i in range(num_total_tokens - block_size):
    # Extract the input sequence chunk
    x_chunk = full_data_sequence[i : i + block_size]
    # Extract the target sequence chunk (shifted one position right)
    y_chunk = full_data_sequence[i + 1 : i + block_size + 1]
    all_x.append(x_chunk)
    all_y.append(y_chunk)

# Stack the lists of tensors into single large tensors
train_x = torch.stack(all_x)
train_y = torch.stack(all_y)

num_sequences_available = train_x.shape[0]
print(f"Created {num_sequences_available} overlapping input/target sequence pairs.")
print(f"Shape of train_x: {train_x.shape}") # Should be (num_sequences, block_size)
print(f"Shape of train_y: {train_y.shape}") # Should be (num_sequences, block_size)

# Optional: Verify device
# print(f"train_x is on device: {train_x.device}") # May still be on CPU, move in batching



### OUTPUT ###
# Created 529 overlapping input/target sequence pairs.
# Shape of train_x: torch.Size([529, 64])
# Shape of train_y: torch.Size([529, 64])
```

From our 593-character text, we were able to extract 529 overlapping sequences of length `64` (`block_size`).

The output confirms this, showing that `train_x` (inputs) and `train_y` (targets) are now tensors with 529 rows (sequences) and 64 columns (token IDs per sequence).

Notice that these tensors might still be on the `CPU`; we'll move individual batches to the `GPU` (`device`) during training.

## Batching Strategy (Random Sampling)

Training on the entire dataset at once is usually too memory-intensive. Instead, we train in `mini-batches`.

A common strategy, and the one we'll use here for simplicity, is `random sampling`. In each training step, we'll randomly pick `batch_size` indices (from `0` to `num_sequences_available` - 1) and grab the corresponding input/target pairs from `train_x` and `train_y`.

These selected batches will then be moved to the `device` (`GPU` or `CPU`) for processing by the model.

```python
# Check if we have enough sequences for the desired batch size
if num_sequences_available < batch_size:
    print(f"Warning: Number of sequences ({num_sequences_available}) is less than batch size ({batch_size}). Adjusting batch size.")
    batch_size = num_sequences_available

print(f"Data ready for training. Will sample batches of size {batch_size} randomly.")
print("Batches will be moved to device during the training loop.")
# Example of how a batch would be selected in the loop:
# indices = torch.randint(0, num_sequences_available, (batch_size,))
# xb = train_x[indices].to(device)
# yb = train_y[indices].to(device)


### OUTPUT ###
# Data ready for training. Will sample batches of size 16 randomly.
# Batches will be moved to device during the training loop.
```

This confirms our plan. We have enough sequences (529) for our chosen batch_size (16). It reminds us that in each training step, we will randomly grab 16 input/target sequence pairs and send them to the GPU/CPU for that step’s calculation.

## Model Component Initialization

This is the first layer of the model. It takes the integer token IDs (like the ones in `train_x`) and converts each one into a dense vector of size `d_model`. Think of it as a lookup table where each token ID has its own unique vector representation.

These vectors capture some initial "meaning" of the tokens, which the model will learn and refine during training.

Input shape: `(Batch, SequenceLength)` → Output shape: `(Batch, SequenceLength, d_model)`.

![MC Initialization](https://miro.medium.com/v2/resize:fit:875/1*OODMy-kyg-eTvo7HdCe9nQ.png)

```python
# Initialize the token embedding table
token_embedding_table = nn.Embedding(vocab_size, d_model).to(device)

print(f"Initialized Token Embedding Layer:")
print(f"  Input Vocab Size: {vocab_size}")
print(f"  Output Embedding Dim (d_model): {d_model}")
print(f"  Weight shape: {token_embedding_table.weight.shape}")
print(f"  Device: {token_embedding_table.weight.device}")


### OUTPUT ###
# Initialized Token Embedding Layer:
#   Input Vocab Size: 36
#   Output Embedding Dim (d_model): 128
#   Weight shape: torch.Size([36, 128])
#   Device: cuda:0
```

We’ve created the `nn.Embedding` layer. The output shows it's configured correctly: it knows our `vocab_size` is `36` and will output vectors of size `d_model` (`128`).

The `Weight` shape confirms the lookup table size: `36` rows (one for each character) and `128` columns (the embedding dimension). It's also placed on our `GPU` (`cuda:0`).

## Rotary Positional Embedding (RoPE) Precomputation

Transformers don’t inherently understand word order. Positional encodings add this information.

![RoPE Mecahnism](https://miro.medium.com/v2/resize:fit:875/1*i9AEBWXDAg7SD2toXLnkog.png)

RoPE is a clever method used in models like LLaMA. Instead of adding separate position vectors, it *rotates* parts of the Query (Q) and Key (K) vectors in the attention mechanism based on their position.

The amount of rotation depends on the position and pre-calculated frequencies derived from the rope_theta hyperparameter. Here, we precompute the inverse frequencies (inv_freq) which are constant.

The actual rotations (using complex numbers freqs_cis) will be calculated dynamically for each sequence length during the forward pass.

```python
# Precompute the inverse frequencies for RoPE
# Formula: 1.0 / (rope_theta ** (torch.arange(0, d_k, 2) / d_k))
rope_freq_indices = torch.arange(0, d_k, 2, dtype=torch.float, device=device)
inv_freq = 1.0 / (rope_theta ** (rope_freq_indices / d_k))

print("Precomputed RoPE inverse frequencies (inv_freq):")
print(f"  Shape: {inv_freq.shape}") # Should be (d_k / 2,)
print(f"  Values (first 5): {inv_freq[:5].tolist()}")
print(f"  Device: {inv_freq.device}")
# The 'freqs_cis' (complex numbers) will be computed in the forward pass using these inv_freq and position_ids


### OUTPUT ###
# Precomputed RoPE inverse frequencies (inv_freq):
#   Shape: torch.Size([16])
#   Values (first 5): [1.0, 0.5623413324356079, 0.3162277638912201, 0.17782793939113617, 0.10000000149011612]
#   Device: cuda:0
```

This block calculates and stores the `inv_freq` tensor. Since our dimension per head (`d_k`) is `32`, `RoPE` works on pairs, so the shape is `(16,)` (i.e., `d_k / 2`).

These values represent the base frequencies for the rotations. We'll use this `inv_freq` tensor later in the forward pass to calculate the actual rotation angles (`freqs_cis`) based on the position of each token.

## RMSNorm Layers Initialization

Normalization layers help stabilize training. LLaMA uses RMSNorm (Root Mean Square Normalization), which is simpler and faster than standard Layer Normalization.

It normalizes the input vector by its root-mean-square value and then scales it using a learnable parameter gamma (weight). We don’t usually have a learnable bias (beta) like in LayerNorm.

We need RMSNorm before the attention block and before the MoE/FFN block in each layer, plus a final one before the output layer.

Since we’re doing this inline, we’ll just initialize the learnable gamma weights (nn.Parameter) here; the actual RMS calculation will happen in the forward pass.

![RMSNorm Layers Initialization](https://miro.medium.com/v2/resize:fit:875/1*aKfU3MA05w9MK783w5JrIw.png)

```python
# Lists to store RMSNorm layer weights for each Transformer block
rmsnorm_weights_input = []      # RMSNorm before MHA
rmsnorm_weights_post_attn = []  # RMSNorm before MoE/FFN

print(f"Initializing RMSNorm weights for {n_layers} layers...")
for i in range(n_layers):
    # RMSNorm weight for input to attention
    # Initialize weight as torch.ones, similar to nn.LayerNorm's default gamma
    weight_in = nn.Parameter(torch.ones(d_model, device=device))
    rmsnorm_weights_input.append(weight_in)

    # RMSNorm weight for input to MoE/FFN (post-attention)
    weight_post = nn.Parameter(torch.ones(d_model, device=device))
    rmsnorm_weights_post_attn.append(weight_post)
    print(f"  Initialized RMSNorm weights for Layer {i+1} (Input: {weight_in.shape}, PostAttn: {weight_post.shape})")

# Final RMSNorm before the output layer
final_rmsnorm_weight = nn.Parameter(torch.ones(d_model, device=device))

print(f"Initialized Final RMSNorm weight, shape: {final_rmsnorm_weight.shape}")
print("RMSNorm weights initialized (as nn.Parameter). The normalization logic will be inline.")



### OUTPUT ###
# Initializing RMSNorm weights for 4 layers...
#   Initialized RMSNorm weights for Layer 1 (Input: torch.Size([128]), PostAttn: torch.Size([128]))
#   Initialized RMSNorm weights for Layer 2 (Input: torch.Size([128]), PostAttn: torch.Size([128]))
#   Initialized RMSNorm weights for Layer 3 (Input: torch.Size([128]), PostAttn: torch.Size([128]))
#   Initialized RMSNorm weights for Layer 4 (Input: torch.Size([128]), PostAttn: torch.Size([128]))
# Initialized Final RMSNorm weight, shape: torch.Size([128])
# RMSNorm weights initialized (as nn.Parameter). The normalization logic will be inline.
```

Here, we created the learnable `gamma` weights for all the `RMSNorm` operations needed. For each of our `n_layers` (`4` layers), we need one weight before attention (`rmsnorm_weights_input`) and one before the `MoE` block (`rmsnorm_weights_post_attn`).

We also need one final weight (`final_rmsnorm_weight`) after the last layer. Each weight is a `Parameter` tensor of size `d_model` (`128`), initialized to ones. The actual math for `RMSNorm` will use these weights during the forward pass.

#### Attention Layers Initialization (MHA)

The core of the Transformer is the self-attention mechanism. We’re using Multi-Head Attention (MHA).

For each layer, we need linear projection layers to transform the input vectors into Query (Q), Key (K), and Value (V) spaces.

1.  A `QKV Projection` is single large linear layer takes the input (size `d_model`) and projects it to a combined QKV space (size `3 * d_model`).
2.  `Output Projection`: After attention is calculated using `Q`, `K`, and `V` across multiple heads, another linear layer projects the combined result back to the original `d_model` dimension.

We'll initialize these `nn.Linear` layers for each Transformer block. Often, bias is turned off in these projections in large models.

![Multi Head Attention](https://miro.medium.com/v2/resize:fit:1250/1*S8G50sCytOE8NAoXI_vhDw.png)

```python
# Lists to store Attention layers for each Transformer block
mha_qkv_linears = []    # Combined Linear layer for Q, K, V projections
mha_output_linears = [] # Output Linear layer for MHA

print(f"Initializing Attention (MHA) linear layers for {n_layers} layers...")
for i in range(n_layers):
    # Combined QKV projection layer
    # Bias is often False in large transformer QKV projections
    qkv_linear = nn.Linear(d_model, 3 * d_model, bias=False).to(device)
    mha_qkv_linears.append(qkv_linear)

    # Output projection layer
    # Bias is often False here too, but can be True
    output_linear = nn.Linear(d_model, d_model, bias=False).to(device)
    mha_output_linears.append(output_linear)
    print(f"  Initialized MHA Linears for Layer {i+1} (QKV: {qkv_linear.weight.shape}, Out: {output_linear.weight.shape})")

print("Attention (MHA) linear layers initialized.")


### OUTPUT ###
# Initializing Attention (MHA) linear layers for 4 layers...
#   Initialized MHA Linears for Layer 1 (QKV: torch.Size([384, 128]), Out: torch.Size([128, 128]))
#   Initialized MHA Linears for Layer 2 (QKV: torch.Size([384, 128]), Out: torch.Size([128, 128]))
#   Initialized MHA Linears for Layer 3 (QKV: torch.Size([384, 128]), Out: torch.Size([128, 128]))
#   Initialized MHA Linears for Layer 4 (QKV: torch.Size([384, 128]), Out: torch.Size([128, 128]))
# Attention (MHA) linear layers initialized.
```

This sets up the linear layers needed for attention in each of our 4 Transformer blocks. For each layer, we have:

*   `qkv_linear`: A layer mapping `d_model` (`128`) to `3 * d_model` (`384`). Its weight shape is `[384, 128]`.
*   `output_linear`: A layer mapping `d_model` (`128`) back to `d_model` (`128`). Its weight shape is `[128, 128]`.

These layers are stored in lists (`mha_qkv_linears`, `mha_output_linears`) so we can access the correct one for each layer during the forward pass.

## Mixture-of-Experts (MoE) Layers Initialization

This is the special part. Instead of a single large Feed-Forward Network (FFN) after the attention block, we use an MoE layer. For each layer, this involves:

![MoE Layers](https://miro.medium.com/v2/resize:fit:875/1*DmDavR13bD_EtyDlwFv7uQ.png)

*   `Router`: A simple linear layer that takes the token's hidden state (size `d_model`) and outputs a score (logit) for each available "expert".
*   `Experts`: A collection (`num_local_experts`) of smaller, independent MLPs. Each expert is typically a "gated MLP", similar to the standard FFN in `LLaMA`: it has parallel "gate" and "up" projections, followed by an activation (`SiLU`/`Swish`), multiplication (gating), and a "down" projection.
*   We initialize the weights for all experts. We'll store these expert weights directly as `nn.Parameter` tensors rather than lists of `nn.Linear` layers.
*   `Shared Expert`: A standard gated MLP (just like one of the experts) that all tokens pass through. Its output is added to the combined output of the selected experts.

The `router` decides which `num_experts_per_tok` experts each token should go to (Top-K routing). The outputs of these selected experts are then combined, weighted by the router's confidence scores.

```python
# Lists to store MoE components for each layer
moe_routers = []             # Router linear layers
moe_expert_gate_up_proj = [] # Expert Gate/Up projection weights
moe_expert_down_proj = []    # Expert Down projection weights
shared_expert_gate_proj = [] # Shared Expert Gate projection
shared_expert_up_proj = []   # Shared Expert Up projection
shared_expert_down_proj = [] # Shared Expert Down projection

print(f"Initializing MoE and Shared MLP components for {n_layers} layers...")
print(f"  Num Experts per layer: {num_local_experts}")
print(f"  Expert Dim: {expert_dim}")
print(f"  Shared MLP Dim: {shared_expert_dim}")

for i in range(n_layers):
    # 1. Router
    router_linear = nn.Linear(d_model, num_local_experts, bias=False).to(device)
    moe_routers.append(router_linear)

    # 2. Experts (Weights as Parameters)
    # Gate/Up Projection Weight: (num_experts, d_model, 2 * expert_dim)
    # Note: Combining Gate and Up projection into one weight matrix here
    gate_up_w = nn.Parameter(torch.empty(num_local_experts, d_model, 2 * expert_dim, device=device))
    nn.init.normal_(gate_up_w, mean=0.0, std=0.02) # Example initialization
    moe_expert_gate_up_proj.append(gate_up_w)

    # Down Projection Weight: (num_experts, expert_dim, d_model)
    down_w = nn.Parameter(torch.empty(num_local_experts, expert_dim, d_model, device=device))
    nn.init.normal_(down_w, mean=0.0, std=0.02) # Example initialization
    moe_expert_down_proj.append(down_w)

    # 3. Shared Expert (Standard MLP layers)
    shared_gate = nn.Linear(d_model, shared_expert_dim, bias=False).to(device)
    shared_up = nn.Linear(d_model, shared_expert_dim, bias=False).to(device)
    shared_down = nn.Linear(shared_expert_dim, d_model, bias=False).to(device)
    shared_expert_gate_proj.append(shared_gate)
    shared_expert_up_proj.append(shared_up)
    shared_expert_down_proj.append(shared_down)

    print(f"  Initialized MoE components for Layer {i+1}:")
    print(f"    Router weights: {router_linear.weight.shape}")
    print(f"    Expert Gate/Up weights: {gate_up_w.shape}")
    print(f"    Expert Down weights: {down_w.shape}")
    print(f"    Shared Gate weights: {shared_gate.weight.shape}")
    print(f"    Shared Up weights: {shared_up.weight.shape}")
    print(f"    Shared Down weights: {shared_down.weight.shape}")

print("MoE and Shared MLP components initialized.")
# Activation function (used inline)
activation_fn = nn.SiLU()
```

This output shows the initialization for the MoE components in each of our 4 layers. For each layer, we created:

```
Initializing MoE and Shared MLP components for 4 layers...
  Num Experts per layer: 4
  Expert Dim: 256
  Shared MLP Dim: 256
  Initialized MoE components for Layer 1:
    Router weights: torch.Size([4, 128])
    Expert Gate/Up weights: torch.Size([4, 128, 512]) # num_experts, d_model, 2*expert_dim
    Expert Down weights: torch.Size([4, 256, 128])  # num_experts, expert_dim, d_model
    Shared Gate weights: torch.Size([256, 128])
    Shared Up weights: torch.Size([256, 128])
    Shared Down weights: torch.Size([128, 256])
  ... (similar output for Layers 2, 3, 4) ...
MoE and Shared MLP components initialized.
```

*   `Router weights`: A linear layer mapping `d_model` (`128`) to the number of experts (`4`). Shape `[4, 128]`.
*   `Expert Gate/Up weights`: A single parameter tensor holding the combined gate and up projection weights for all 4 experts. Shape `[num_experts, d_model, 2 * expert_dim] = [4, 128, 512]`.
*   `Expert Down weights`: A parameter tensor holding the down projection weights for all 4 experts. Shape `[num_experts, expert_dim, d_model] = [4, 256, 128]`.
*   `Shared Gate/Up/Down weights`: Standard linear layers for the shared expert MLP, with shapes corresponding to `d_model` (`128`) and `shared_expert_dim` (`256`).

These components are stored in lists, ready for the complex `MoE` logic in the forward pass. We also define the `SiLU` activation function.

## Final Output Layer Initialization

After the input has passed through all the Transformer layers, the final hidden state (after one last RMSNorm) needs to be converted into predictions for the next token.

This final linear layer takes the d_model-sized vector for each position and projects it to a vector of size vocab_size.

Each element in this output vector represents the raw score (logit) for a potential next character in our vocabulary.

![Output layer](https://miro.medium.com/v2/resize:fit:875/1*fZjQtnZJgWmah3GaJVY60g.png)

```python
# Final Linear Layer (language modeling head)
output_linear_layer = nn.Linear(d_model, vocab_size, bias=False).to(device)

print(f"Initialized Final Output Linear Layer:")
print(f"  Input Dim (d_model): {d_model}")
print(f"  Output Dim (vocab_size): {vocab_size}")
print(f"  Weight shape: {output_linear_layer.weight.shape}")
print(f"  Device: {output_linear_layer.weight.device}")


### OUTPUT ###
# Initialized Final Output Linear Layer:
#   Input Dim (d_model): 128
#   Output Dim (vocab_size): 36
#   Weight shape: torch.Size([36, 128])
#   Device: cuda:0
```

We initialize the final `nn.Linear` layer. It takes the `d_model` (`128`) dimension as input and outputs `vocab_size` (`36`) logits. The weight shape `[36, 128]` confirms this mapping.

## Causal Mask Precomputation

In a decoder-only Transformer like this, when predicting the token at position `t`, the model should only attend to tokens at positions `0` to `t` (including itself) and not to future tokens (`t+1`, `t+2`, ...).

The `causal mask` enforces this. It's a matrix used during the attention calculation. We create a lower triangular matrix (size `block_size x block_size`) where positions the model can attend to have a value (like `1`) and positions it cannot attend to have another value (like `0`).

This mask is applied before the `softmax` step in attention, effectively setting the scores for future positions to negative infinity. We precompute this for the maximum sequence length (`block_size`).

```python
# Create the lower triangular mask for causal self-attention
# Values are 1 where attention is allowed, 0 where it's masked.
# Shape: (1, 1, block_size, block_size) for broadcasting with (B, n_heads, T, T)
causal_mask = torch.tril(torch.ones(block_size, block_size, device=device))
causal_mask = causal_mask.view(1, 1, block_size, block_size)

print("Precomputed Causal Attention Mask:")
print(f"  Shape: {causal_mask.shape}")
print(f"  Requires grad: {causal_mask.requires_grad}")
# Optional: Visualize the mask for a smaller block size
# if block_size <= 8:
#    print(causal_mask[0, 0].cpu().numpy())


### OUTPUT ###
# Precomputed Causal Attention Mask:
#   Shape: torch.Size([1, 1, 64, 64])
#   Requires grad: False
```

This creates the `causal_mask`. It's a tensor filled with `1`s in the lower triangle (including the diagonal) and `0`s elsewhere.

The shape `[1, 1, 64, 64]` is set up for easy broadcasting with the attention scores tensor (which has shape `[Batch, n_heads, SeqLen, SeqLen]`) during the forward pass. It doesn't require gradients because it's fixed.

## Training Setup

The `optimizer` is the algorithm that updates the model's weights based on the gradients calculated during backpropagation (learning). We'll use `AdamW`, a popular and effective optimizer for Transformers.

To use it, we first need to collect all the parameters in our model that need to be trained (i.e., have `requires_grad=True`).

This includes the weights of the `embedding` table, all the `linear layers` (`QKV`, `output`, `MoE` routers, shared experts), and the `nn.Parameter` tensors we created for `RMSNorm` weights and `MoE` expert weights.

```python
# Gather all model parameters requiring gradients
all_model_parameters = list(token_embedding_table.parameters())
# Add RMSNorm weights
all_model_parameters.extend(rmsnorm_weights_input)
all_model_parameters.extend(rmsnorm_weights_post_attn)
all_model_parameters.append(final_rmsnorm_weight)
# Add Attention linear layer weights
for i in range(n_layers):
    all_model_parameters.extend(list(mha_qkv_linears[i].parameters()))
    all_model_parameters.extend(list(mha_output_linears[i].parameters()))
# Add MoE Router linear layer weights
for i in range(n_layers):
    all_model_parameters.extend(list(moe_routers[i].parameters()))
# Add MoE Expert weights (already nn.Parameters)
all_model_parameters.extend(moe_expert_gate_up_proj)
all_model_parameters.extend(moe_expert_down_proj)
# Add Shared Expert linear layer weights
for i in range(n_layers):
    all_model_parameters.extend(list(shared_expert_gate_proj[i].parameters()))
    all_model_parameters.extend(list(shared_expert_up_proj[i].parameters()))
    all_model_parameters.extend(list(shared_expert_down_proj[i].parameters()))
# Add Final Output linear layer weights
all_model_parameters.extend(list(output_linear_layer.parameters()))

# Count total number of parameter tensors (groups)
num_param_groups = len(all_model_parameters)
# Count total number of individual parameters
total_params = sum(p.numel() for p in all_model_parameters if p.requires_grad)

# Define the AdamW optimizer
optimizer = optim.AdamW(all_model_parameters, lr=learning_rate)

print("Optimizer Setup:")
print(f"  Optimizer: {type(optimizer).__name__}")
print(f"  Learning Rate: {learning_rate}")
print(f"  Managing {num_param_groups} parameter groups/tensors.")
print(f"  Total Trainable Parameters: {total_params:,}")



#### OUTPUT ####
# Optimizer Setup:
#   Optimizer: AdamW
#   Learning Rate: 0.0005
#   Managing 43 parameter groups/tensors.
#   Total Trainable Parameters: 2,240,640
```

The code successfully gathered all the trainable parts of our model (`43` distinct weight/bias tensors or parameter objects) and created the `AdamW` optimizer to manage them using our specified `learning_rate`.

It also calculated the total number of individual trainable parameters in our model, which is about `2.24 million` – tiny compared to real models.

## Define Loss Function

We need a way to measure how “wrong” the model’s predictions are compared to the actual target tokens. Since predicting the next token is a classification problem (choosing the correct character from our vocabulary), the standard loss function is Cross-Entropy Loss.

It takes the model’s output logits and the true target token IDs and calculates a score representing the error.

```python
# Define the loss function
criterion = nn.CrossEntropyLoss()
```

We’ve initialized the `nn.CrossEntropyLoss` function. This `criterion` object will be used inside the training loop to compute the loss value for each batch.

## Training the Model

We will iteratively train the model by feeding it batches of data, calculating the loss, and updating the parameters using the optimizer.

This is where all the previously initialized components come together in the forward pass.

For a set number of epochs, we repeat the following:

![Training loop pass](https://miro.medium.com/v2/resize:fit:1250/1*LhLkclDBw2vpyADqZFa-8A.png)

```python
print(f"\n--- Starting Training Loop for {epochs} epochs ---")

losses = []

for epoch in range(epochs):
    # Sample a random batch of data
    ix = torch.randint(num_sequences_available, (batch_size,))
    xb = train_x[ix].to(device)
    yb = train_y[ix].to(device)

    # --- Forward Pass ---
    B, T = xb.shape
    token_embed = token_embedding_table(xb) # (B, T, d_model)

    # Prepare RoPE frequencies for the current sequence length
    position_ids = torch.arange(T, device=device).unsqueeze(0) # (1, T)
    # Correct calculation for freqs_cis
    t_indices = torch.arange(T, device=device)
    freqs = 1.0 / (rope_theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)) # (d_k/2)
    m_theta = torch.outer(t_indices, freqs).float() # (T, d_k/2)
    freqs_cis = torch.polar(torch.ones_like(m_theta), m_theta) # (T, d_k/2) complex
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # (1, T, 1, d_k/2) for broadcasting


    x = token_embed
    for i in range(n_layers):
        # Residual connection starts here
        residual = x

        # 1. RMSNorm before Attention
        x_norm = (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_norm_eps)) * rmsnorm_weights_input[i]

        # 2. Multi-Head Attention
        qkv = mha_qkv_linears[i](x_norm) # (B, T, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each is (B, T, d_model)

        # Reshape for multi-head
        q = q.view(B, T, n_heads, d_k) # (B, T, n_heads, d_k)
        k = k.view(B, T, n_heads, d_k)
        v = v.view(B, T, n_heads, d_k)

        # Apply RoPE to Q and K
        q = q.view(B, T, n_heads, d_k//2, 2)
        k = k.view(B, T, n_heads, d_k//2, 2)
        q_complex = torch.view_as_complex(q.float()) # (B, T, n_heads, d_k/2)
        k_complex = torch.view_as_complex(k.float()) # (B, T, n_heads, d_k/2)

        # Apply rotation based on position
        # Ensure freqs_cis aligns with sequence length T
        q_rotated_complex = q_complex * freqs_cis[:, :T] # freqs_cis needs proper shape (1, T, 1, d_k/2)
        k_rotated_complex = k_complex * freqs_cis[:, :T]

        q_rotated = torch.view_as_real(q_rotated_complex).view(B, T, n_heads, d_k)
        k_rotated = torch.view_as_real(k_rotated_complex).view(B, T, n_heads, d_k)


        # Transpose for attention calculation: (B, n_heads, T, d_k)
        q_rotated = q_rotated.transpose(1, 2)
        k_rotated = k_rotated.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = (q_rotated @ k_rotated.transpose(-2, -1)) * (d_k ** -0.5) # (B, n_heads, T, T)
        # Apply causal mask
        attn_scores = attn_scores.masked_fill(causal_mask[:,:,:T,:T] == 0, float('-inf'))
        attention_weights = F.softmax(attn_scores, dim=-1) # (B, n_heads, T, T)
        attn_output = attention_weights @ v # (B, n_heads, T, d_k)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, d_model) # (B, T, d_model)
        attn_output = mha_output_linears[i](attn_output)

        # Add attention output to residual
        x = residual + attn_output

        # 3. RMSNorm before MoE/FFN & Residual for MoE/FFN
        residual_moe = x
        x_norm_moe = (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_norm_eps)) * rmsnorm_weights_post_attn[i]

        # 4. MoE Block
        router_logits = moe_routers[i](x_norm_moe) # (B, T, num_local_experts)
        routing_weights, selected_experts = torch.topk(router_logits, num_experts_per_tok, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1).to(x_norm_moe.dtype) # Normalize scores per token

        # Initialize final output tensor
        final_hidden_state = torch.zeros_like(x_norm_moe)

        # Flatten tokens and experts for batch processing
        flat_x = x_norm_moe.view(-1, d_model)                     # (B*T, d_model)
        flat_router_weights = routing_weights.view(-1, num_experts_per_tok) # (B*T, num_experts_per_tok)
        flat_selected_experts = selected_experts.view(-1, num_experts_per_tok) # (B*T, num_experts_per_tok)

        # Calculate expert outputs
        expert_outputs_list = []
        for k in range(num_experts_per_tok):
            expert_idx = flat_selected_experts[:, k] # Indices of the k-th best expert for each token (B*T)
            token_indices = torch.arange(flat_x.size(0), device=device)

            # Get weights for the selected experts
            gate_up_w_k = moe_expert_gate_up_proj[i][expert_idx] # (B*T, d_model, 2 * expert_dim)
            down_w_k = moe_expert_down_proj[i][expert_idx]     # (B*T, expert_dim, d_model)

            # Perform expert calculations using bmm
            # Input needs shape (B*T, 1, d_model) for bmm with (B*T, d_model, 2*expert_dim)
            expert_input_k = flat_x.unsqueeze(1) # (B*T, 1, d_model)
            gate_up_out_k = torch.bmm(expert_input_k, gate_up_w_k) # (B*T, 1, 2 * expert_dim)

            # Split gate and up projections
            gate_k, up_k = gate_up_out_k.chunk(2, dim=-1) # Each (B*T, 1, expert_dim)

            # Apply activation and gating
            activated_up_k = activation_fn(gate_k) * up_k # (B*T, 1, expert_dim)

            # Down projection
            # Input needs shape (B*T, 1, expert_dim) for bmm with (B*T, expert_dim, d_model)
            expert_output_k = torch.bmm(activated_up_k, down_w_k) # (B*T, 1, d_model)
            expert_output_k = expert_output_k.squeeze(1) # (B*T, d_model)

            # Weight the expert output
            expert_output_weighted_k = expert_output_k * flat_router_weights[:, k].unsqueeze(1)
            expert_outputs_list.append(expert_output_weighted_k)

        # Sum the weighted outputs of the selected experts
        moe_output = torch.stack(expert_outputs_list, dim=0).sum(dim=0) # Sum over num_experts_per_tok
        moe_output = moe_output.view(B, T, d_model)

        # 5. Shared Expert MLP (applied to the same x_norm_moe)
        shared_gate_val = shared_expert_gate_proj[i](x_norm_moe)
        shared_up_val = shared_expert_up_proj[i](x_norm_moe)
        shared_output = shared_expert_down_proj[i](activation_fn(shared_gate_val) * shared_up_val)

        # Add MoE output and Shared output to the residual
        x = residual_moe + moe_output + shared_output

    # --- Final Layer ---
    # RMSNorm before final layer
    x = (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + rms_norm_eps)) * final_rmsnorm_weight
    logits = output_linear_layer(x) # (B, T, vocab_size)

    # --- Calculate Loss ---
    # Reshape logits and targets for CrossEntropyLoss
    loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))

    # --- Backward Pass and Optimization ---
    optimizer.zero_grad(set_to_none=True) # More efficient zeroing
    loss.backward()
    optimizer.step()

    # --- Logging ---
    losses.append(loss.item())
    if epoch % eval_interval == 0 or epoch == epochs - 1:
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("--- Training Loop Completed ---")

try:
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
except ImportError:
    print("Matplotlib not found, skipping loss plot.")
```

When we start the training, It will start printing the training loss.

```
--- Starting Training Loop for 3000 epochs ---
  Epoch 1/3000, Loss: 3.8124
  Epoch 301/3000, Loss: 0.0734
  Epoch 601/3000, Loss: 0.0595
  Epoch 901/3000, Loss: 0.0609
  Epoch 1201/3000, Loss: 0.0707
  Epoch 1501/3000, Loss: 0.0664
  Epoch 1801/3000, Loss: 0.0559
  Epoch 2101/3000, Loss: 0.0610
  Epoch 2401/3000, Loss: 0.0680
  Epoch 2701/3000, Loss: 0.0641
  Epoch 3000/3000, Loss: 0.0553
--- Training Loop Completed ---
```

![Training loss](https://miro.medium.com/v2/resize:fit:875/1*4qRaX8cpCL7uoj7TZWWksQ.png)
*Training loss*

The output shows the training progress. The loss starts relatively high (around 3.8) and decreases significantly over the 3000 epochs, settling around 0.05–0.07.

This sharp drop is exactly what we want! It means the model is learning the patterns in the “Alice in Wonderland” text and getting much better at predicting the next character.

The plot visually confirms this downward trend in the loss. The MoE layers, RMSNorm, and RoPE are working together.

## Text Generation

Now that the model is trained, let’s see what it can write! We start with a short prompt (seed text). We convert this prompt into token IDs.

We also specify how many new tokens (characters) we want the model to generate. It’s important to set the model components to “evaluation mode” (using `.eval()`).

This turns off things like dropout if we had used them, ensuring consistent output. We also use `torch.no_grad()` because we're not training anymore, so we don't need PyTorch to track gradients, making generation faster and using less memory.

```python
print("\n--- Step 7: Text Generation ---")

# --- Generation Parameters ---
seed_chars = "Alice " # Starting text prompt
num_tokens_to_generate = 200 # How many new characters to generate
print(f"Seed text: '{seed_chars}'")
print(f"Generating {num_tokens_to_generate} new tokens...")

# --- Prepare Initial Context ---
# Convert seed characters to token IDs
seed_ids = [char_to_int[ch] for ch in seed_chars if ch in char_to_int]
# Create the initial context tensor (add batch dimension)
generated_sequence = torch.tensor([seed_ids], dtype=torch.long, device=device)
print(f"Initial context shape: {generated_sequence.shape}")

# --- Set Model Components to Evaluation Mode ---
# (Important if Dropout or BatchNorm were used, good practice anyway)
token_embedding_table.eval()
for i in range(n_layers):
    # RMSNorm doesn't have eval mode, just use weights
    mha_qkv_linears[i].eval()
    mha_output_linears[i].eval()
    moe_routers[i].eval()
    # Expert weights (Parameters) don't have eval()
    shared_expert_gate_proj[i].eval()
    shared_expert_up_proj[i].eval()
    shared_expert_down_proj[i].eval()
output_linear_layer.eval()
# Final RMSNorm weight doesn't have eval()
print("Model components set to evaluation mode (where applicable).")


### OUTPUT ###
# --- Step 7: Text Generation ---
# Seed text: 'Alice '
# Generating 200 new tokens...
# Initial context shape: torch.Size([1, 6])
# Model components set to evaluation mode (where applicable).
```

This sets up the generation process. Our starting prompt is `"Alice "`. We aim to generate `200` more characters. The initial prompt is converted to a tensor of token IDs with shape `[1, 6]` (1 sequence in the batch, 6 tokens long). The relevant model layers are switched to evaluation mode.

## The Generation Loop

We generate text one character at a time in a loop:

![Generation Loop](https://miro.medium.com/v2/resize:fit:1250/1*77KdQIJ5qrkW47GxSWbivw.png)

```python
print("Starting generation loop...")

with torch.no_grad():
    for _ in range(num_tokens_to_generate):
        # Ensure context doesn't exceed block_size
        current_context = generated_sequence[:, -block_size:]
        B_gen, T_gen = current_context.shape

        # --- Forward pass (similar to training, but without loss calc) ---
        token_embed_gen = token_embedding_table(current_context)

        # Prepare RoPE frequencies for the current sequence length T_gen
        t_indices_gen = torch.arange(T_gen, device=device)
        freqs_gen = 1.0 / (rope_theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k)) # (d_k/2)
        m_theta_gen = torch.outer(t_indices_gen, freqs_gen).float() # (T_gen, d_k/2)
        freqs_cis_gen = torch.polar(torch.ones_like(m_theta_gen), m_theta_gen) # (T_gen, d_k/2) complex
        freqs_cis_gen = freqs_cis_gen.unsqueeze(0).unsqueeze(2) # (1, T_gen, 1, d_k/2)

        x_gen = token_embed_gen
        for i in range(n_layers):
            residual_gen = x_gen
            x_norm_gen = (x_gen.float() * torch.rsqrt(x_gen.pow(2).mean(-1, keepdim=True) + rms_norm_eps)) * rmsnorm_weights_input[i]

            qkv_gen = mha_qkv_linears[i](x_norm_gen)
            q_gen, k_gen, v_gen = qkv_gen.chunk(3, dim=-1)

            q_gen = q_gen.view(B_gen, T_gen, n_heads, d_k)
            k_gen = k_gen.view(B_gen, T_gen, n_heads, d_k)
            v_gen = v_gen.view(B_gen, T_gen, n_heads, d_k)

            # Apply RoPE
            q_gen = q_gen.view(B_gen, T_gen, n_heads, d_k//2, 2)
            k_gen = k_gen.view(B_gen, T_gen, n_heads, d_k//2, 2)
            q_complex_gen = torch.view_as_complex(q_gen.float())
            k_complex_gen = torch.view_as_complex(k_gen.float())
            q_rotated_complex_gen = q_complex_gen * freqs_cis_gen # Use freqs_cis_gen
            k_rotated_complex_gen = k_complex_gen * freqs_cis_gen # Use freqs_cis_gen
            q_rotated_gen = torch.view_as_real(q_rotated_complex_gen).view(B_gen, T_gen, n_heads, d_k)
            k_rotated_gen = torch.view_as_real(k_rotated_complex_gen).view(B_gen, T_gen, n_heads, d_k)

            q_rotated_gen = q_rotated_gen.transpose(1, 2)
            k_rotated_gen = k_rotated_gen.transpose(1, 2)
            v_gen = v_gen.transpose(1, 2)

            attn_scores_gen = (q_rotated_gen @ k_rotated_gen.transpose(-2, -1)) * (d_k ** -0.5)
            attn_scores_gen = attn_scores_gen.masked_fill(causal_mask[:,:,:T_gen,:T_gen] == 0, float('-inf'))
            attention_weights_gen = F.softmax(attn_scores_gen, dim=-1)
            attn_output_gen = attention_weights_gen @ v_gen
            attn_output_gen = attn_output_gen.transpose(1, 2).contiguous().view(B_gen, T_gen, d_model)
            attn_output_gen = mha_output_linears[i](attn_output_gen)
            x_gen = residual_gen + attn_output_gen

            residual_moe_gen = x_gen
            x_norm_moe_gen = (x_gen.float() * torch.rsqrt(x_gen.pow(2).mean(-1, keepdim=True) + rms_norm_eps)) * rmsnorm_weights_post_attn[i]

            # MoE Block (simplified for generation context)
            router_logits_gen = moe_routers[i](x_norm_moe_gen)
            routing_weights_gen, selected_experts_gen = torch.topk(router_logits_gen, num_experts_per_tok, dim=-1)
            routing_weights_gen = F.softmax(routing_weights_gen, dim=-1).to(x_norm_moe_gen.dtype)

            final_hidden_state_gen = torch.zeros_like(x_norm_moe_gen)
            flat_x_gen = x_norm_moe_gen.view(-1, d_model)
            flat_router_weights_gen = routing_weights_gen.view(-1, num_experts_per_tok)
            flat_selected_experts_gen = selected_experts_gen.view(-1, num_experts_per_tok)

            expert_outputs_list_gen = []
            for k in range(num_experts_per_tok):
                 expert_idx_gen = flat_selected_experts_gen[:, k]
                 gate_up_w_k_gen = moe_expert_gate_up_proj[i][expert_idx_gen]
                 down_w_k_gen = moe_expert_down_proj[i][expert_idx_gen]
                 expert_input_k_gen = flat_x_gen.unsqueeze(1)
                 gate_up_out_k_gen = torch.bmm(expert_input_k_gen, gate_up_w_k_gen)
                 gate_k_gen, up_k_gen = gate_up_out_k_gen.chunk(2, dim=-1)
                 activated_up_k_gen = activation_fn(gate_k_gen) * up_k_gen
                 expert_output_k_gen = torch.bmm(activated_up_k_gen, down_w_k_gen).squeeze(1)
                 expert_output_weighted_k_gen = expert_output_k_gen * flat_router_weights_gen[:, k].unsqueeze(1)
                 expert_outputs_list_gen.append(expert_output_weighted_k_gen)

            moe_output_gen = torch.stack(expert_outputs_list_gen, dim=0).sum(dim=0)
            moe_output_gen = moe_output_gen.view(B_gen, T_gen, d_model)

            shared_gate_val_gen = shared_expert_gate_proj[i](x_norm_moe_gen)
            shared_up_val_gen = shared_expert_up_proj[i](x_norm_moe_gen)
            shared_output_gen = shared_expert_down_proj[i](activation_fn(shared_gate_val_gen) * shared_up_val_gen)

            x_gen = residual_moe_gen + moe_output_gen + shared_output_gen

        # Final Layer prediction
        x_gen = (x_gen.float() * torch.rsqrt(x_gen.pow(2).mean(-1, keepdim=True) + rms_norm_eps)) * final_rmsnorm_weight
        logits_gen = output_linear_layer(x_gen) # (B, T_gen, vocab_size)

        # Focus only on the logits for the last token
        logits_last = logits_gen[:, -1, :] # (B, vocab_size)

        # Apply softmax to get probabilities
        probs = F.softmax(logits_last, dim=-1)

        # Sample the next token ID from the probability distribution
        next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

        # Append the sampled token ID to the sequence
        generated_sequence = torch.cat((generated_sequence, next_token), dim=1)

print("...Generation loop finished.")

```

The generation loop ran for the specified number of steps (200 in our case). Inside the loop (which doesn’t print anything itself), the model repeatedly predicted and appended the next character based on the sequence generated so far.

## Decode Generated Sequence

The generated_sequence tensor now holds the original seed token IDs plus the 200 newly generated token IDs. To see the actual text, we need to convert these numbers back into characters using the int_to_char mapping we created earlier.

We take the sequence of IDs, look up the character for each ID, and join them together into a single string.

```php
# Get the generated sequence for the first (and only) batch item
final_generated_ids = generated_sequence[0].tolist()

# Decode the list of IDs back into a string
decoded_text = ''.join([int_to_char.get(id_val, '[UNK]') for id_val in final_generated_ids])

print("\n--- Final Generated Text ---")
print(decoded_text)


### OUTPUT ###
# --- Final Generated Text ---
# Alice 'without pictures or
# conversation?'
# So she was considering in her own mind (as well as she could, for the
# hot day made her feel very sleepy and stupid), whether the pleasure
# of making a daisy-chain wo ...
```

And here’s the final result! Starting with “Alice “, our trained model generated the next 200 characters. Looking at the output, we can see it has clearly learned the style and content of the training text.

It continues the sentence structure, uses appropriate punctuation, and generates words and phrases directly from the original corpus (“without pictures or conversation?”, “So she was considering…”).

This shows that even our small model with MoE layers successfully learned to predict the next character based on the patterns in the training data.

It’s not generating wildly creative new text (because the training data was tiny and repetitive), but it demonstrates the core generative capability.

## Save Model State (Optional)

After spending time training the model, we usually want to save its state. This involves collecting all the essential information.

```python
# Create a directory to store the model (if it doesn't exist)
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'llama4_moe_model.pt')

# Create a state dictionary manually collecting all components
model_state = {
    # Configuration
    'config': {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'block_size': block_size,
        'rms_norm_eps': rms_norm_eps,
        'rope_theta': rope_theta,
        'num_local_experts': num_local_experts,
        'num_experts_per_tok': num_experts_per_tok,
        'intermediate_size_expert': intermediate_size_expert,
        'intermediate_size_shared': intermediate_size_shared
    },
    # Tokenizer
    'tokenizer': {
        'char_to_int': char_to_int,
        'int_to_char': int_to_char
    },
    # Model Parameters (State Dicts for nn.Modules, Tensors for nn.Parameters)
    'token_embedding_table': token_embedding_table.state_dict(),
    'rmsnorm_weights_input': [p.data for p in rmsnorm_weights_input], # Save tensor data
    'rmsnorm_weights_post_attn': [p.data for p in rmsnorm_weights_post_attn], # Save tensor data
    'final_rmsnorm_weight': final_rmsnorm_weight.data, # Save tensor data
    'mha_qkv_linears': [l.state_dict() for l in mha_qkv_linears],
    'mha_output_linears': [l.state_dict() for l in mha_output_linears],
    'moe_routers': [r.state_dict() for r in moe_routers],
    'moe_expert_gate_up_proj': [p.data for p in moe_expert_gate_up_proj], # Save tensor data
    'moe_expert_down_proj': [p.data for p in moe_expert_down_proj], # Save tensor data
    'shared_expert_gate_proj': [l.state_dict() for l in shared_expert_gate_proj],
    'shared_expert_up_proj': [l.state_dict() for l in shared_expert_up_proj],
    'shared_expert_down_proj': [l.state_dict() for l in shared_expert_down_proj],
    'output_linear_layer': output_linear_layer.state_dict(),
    # Note: RoPE inv_freq is not saved as it's derived from config
}

# Save the state dictionary
torch.save(model_state, save_path)

print(f"Model state saved successfully to '{save_path}'")
```

All the necessary parts of our trained model (configuration, tokenizer, and all the learnable weights) have been packaged into a dictionary and saved into the file `saved_models/llama4_moe_model.pt`.

We could now write separate code to load this file and use the model for generation without needing to rerun the entire training process.

## Conclusion

So, we covered:

1.  **Setup & Tokenization**: Basic environment setup and character-level tokenization.
2.  **Hyperparameter Definition**: Setting up configuration values, scaled down from larger models.
3.  **Data Preparation**: Creating input/target sequences for next-token prediction.
4.  **Model Initialization (Inline)**: Explicitly creating and initializing components like token embeddings, RMSNorm weights, attention linear layers, RoPE frequency bases, MoE routers, MoE expert weights, shared expert MLPs, and the final output layer.
5.  **Training Loop (Inline)**: Implementing the complete forward pass within the loop, demonstrating:

*   Application of RMSNorm.
*   Calculation and application of RoPE within the MHA block.
*   The MoE forward pass: routing, expert selection (Top-K), parallel expert computation (using BMM), combination of expert outputs (scatter_add_), and integration with a shared expert MLP.
*   Standard Transformer operations like residual connections and attention.
*   Loss calculation, backpropagation, and optimizer steps.

6.  **Text Generation**: Implementing autoregressive sampling using the trained model components in evaluation mode.