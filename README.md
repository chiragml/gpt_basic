# 🚀 Mini GPT from Scratch

A minimal yet complete implementation of a GPT-style transformer language model built entirely from scratch using PyTorch. This project demonstrates the core concepts of modern language models, with a focus on self-attention mechanisms and multi-head attention architectures.

## ✨ Features

- **🧠 Self-Attention Mechanism**: Custom implementation of scaled dot-product attention
- **🔄 Multi-Head Attention**: Parallel attention heads for richer representations
- **🏗️ Transformer Architecture**: Complete transformer blocks with layer normalization and residual connections
- **📊 Character-Level Tokenization**: Simple character-based vocabulary for Shakespeare text
- **🎯 Autoregressive Generation**: Text generation with temperature-based sampling
- **📈 Training Pipeline**: Complete training loop with validation loss tracking

## 🏛️ Architecture Overview

```
Input Text → Character Embedding → Positional Embedding
     ↓
Transformer Blocks (3 layers):
├── Layer Normalization
├── Multi-Head Self-Attention (4 heads)
├── Residual Connection
├── Layer Normalization  
├── Feed-Forward Network
└── Residual Connection
     ↓
Layer Normalization → Linear Head → Output Probabilities
```

### 🔍 Key Components

#### Self-Attention Head
```python
class Head(nn.Module):
    '''Single head self-attention mechanism'''
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
```

The attention mechanism computes:
- **Query (Q)**: What information are we looking for?
- **Key (K)**: What information do we have?
- **Value (V)**: The actual information content
- **Attention Weights**: `softmax(QK^T / √d_k)` with causal masking

#### Multi-Head Attention
Multiple attention heads run in parallel, each focusing on different aspects of the relationships between tokens. The outputs are concatenated and projected back to the embedding dimension.

## 📁 Project Structure

```
gpt_basic/
├── bigramv2.py        # 🌟 Main implementation with optimized multi-head attention
├── bigram.py          # Original transformer implementation
├── prepare.py         # Data preprocessing and tokenization
├── input.txt          # Shakespeare dataset
├── requirements.txt   # Dependencies
└── tokens/            # Preprocessed training data
    ├── train.bin
    └── val.bin
```

## 🌟 BigramV2: The Enhanced Implementation

The `bigramv2.py` file contains the optimized version of the transformer where I improved upon my first implementation of multi-head attention for better efficiency and performance.

### 🔧 Multi-Head Attention Evolution

**Original Implementation (`MultiHeadAttention`)**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
```
- Uses individual attention head modules
- Clear separation of concerns
- Easy to understand but less efficient

**Optimized Implementation (`MultiHeadAttn`) - Currently Active**
```python
class MultiHeadAttn(nn.Module):
    def __init__(self, num_heads, n_embd, block_size, dropout):
        # Computes all heads in parallel with tensor reshaping
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
```
- **Batched Computation**: All heads computed simultaneously
- **Memory Efficient**: Single linear transformations instead of multiple
- **Professional Style**: Industry-standard implementation approach

### 🚀 Usage

```bash
# Run the enhanced model
python bigramv2.py
```

**Key Improvements in BigramV2:**
- **Optimized Multi-Head Attention**: Evolved from modular to batched computation
- **Better Performance**: Single matrix operations instead of multiple head modules  
- **Enhanced Tensor Reshaping**: Professional-grade view operations for parallel computation
- **Detailed Documentation**: Better tensor shape annotations throughout

## 🧮 Model Details

### Attention Mechanism

The self-attention mechanism allows each position to attend to all previous positions:

1. **Scaled Dot-Product Attention**:
   ```
   Attention(Q,K,V) = softmax(QK^T / √d_k)V
   ```

2. **Causal Masking**: Lower triangular mask ensures autoregressive property
3. **Dropout**: Applied to attention weights for regularization

### Multi-Head Attention Benefits

- **Parallel Processing**: Multiple attention patterns computed simultaneously
- **Rich Representations**: Different heads can focus on different types of relationships
- **Scalability**: Easily adjustable number of heads for model capacity

## 📊 Performance
Based on the amount of training, the model performs decent. Its able to recognize the structure of English language and tries to build words. Its also able to understand the data's structure and writes in the way that data presents. So I'm happy with the results for now given that the compute cycles were low. 

Sample output:
```
Thy good you deaks; an namech, cannor the cadmaster.

Wich For world of a qur ose, abeins, and to sonst take oph.

ISCAen:
Prened chody dilly words wost, bouse louth sway, leor;
To I maker, ent my hand,s wis sato fears besetale  to at ame.
That'd hope, your make your liangdinveded, tanis take wort.

LUCAPULET:
Beritiolf of litime anstrops as lardeen,
Where the krest word thou was
Aled oup, and hatt fellower.

COTIOLO:
Now stell bissirs, swe same hoold's call not
Hose fame to caus youre Romblees,
```


## 🔬 Experiments

Try modifying these hyperparameters to see their effects:

- **`n_embd`**: Embedding dimension (affects model capacity)
- **`num_heads`**: Number of attention heads (affects representation richness)
- **`num_layers`**: Transformer layers (affects model depth)
- **`block_size`**: Context length (affects memory requirements)
- **`dropout`**: Regularization strength

### Comparing Attention Implementations

The evolution from the original to optimized implementation in `bigramv2.py`:
```python
# Original modular approach (commented out)
# self.sa = MultiHeadAttention(num_heads, head_size)

# Optimized batched approach (currently active)
self.sa = MultiHeadAttn(num_heads, n_embd, block_size, dropout=dropout)
```

**Performance Benefits:**
- **Reduced Memory Allocation**: Single tensor operations vs multiple modules
- **Better GPU Utilization**: Parallel computation across all heads
- **Cleaner Forward Pass**: Streamlined tensor reshaping and operations

## 📚 Learning Resources

This implementation covers key concepts from:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)
- Andrej Karpathy's educational content
---

*Built with ❤️ for understanding transformers from the ground up*