import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 6 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
num_hds = 4
dropout = 0.2
num_layers = 3
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    '''single head self-attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape # B, T, n_embd

        k = self.key(x) # B, T, hd
        q = self.query(x) # B, T, hd

        wei = q @ k.transpose(-2,-1) * C**-0.5 # B, T, T
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1) # B, T, T
        wei = self.drop(wei) # B, T, T
        v = self.value(x) # B, T, hd
        out = wei @ v # (B, T, T) @ (B, T, hd) -> B, T, hd
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        res = torch.cat([head(x) for head in self.heads], dim=-1)
        res = self.proj(res)
        res = self.drop(res)
        return res

class MultiHeadAttn(nn.Module):
    def __init__(self, num_heads, n_embd, block_size, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = n_embd // num_heads
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, D = x.shape

        q = self.query(x) # B, N, D
        k = self.key(x) # B, N, D
        v = self.value(x) # B, N, D

        q = q.view(B, N, self.num_heads, self.head_size).transpose(-3,-2) # B, H, N, Hd
        k = k.view(B, N, self.num_heads, self.head_size).transpose(-3, -2) # B, H, N, Hd
        v = v.view(B, N, self.num_heads, self.head_size).transpose(-3,-2) # B, H, N, Hd

        wei = q @ k.transpose(-2,-1) * D**-0.5  # B, H, N, N
        wei = wei.masked_fill(self.tril[:N,:N] == 0, float('-inf')) # B, H, N, N
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # (B, H, N, N) x (B, H, N, Hd) -> (B, H, N, Hd)
        out = out.transpose(-3, -2).contiguous() # B, N, H, Hd
        out = out.view(B, N, D)
        return out

    


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        # head_size = n_embd // num_heads
        # self.sa = MultiHeadAttention(num_heads, head_size)
        self.sa = MultiHeadAttn(num_heads, n_embd, block_size, dropout=dropout)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # self.as_head = Head(n_embd)
        # self.as_head = MultiHeadAttention(num_hds, n_embd//num_hds)
        # self.ffn = FeedForward(n_embd=n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, num_hds),
        #     Block(n_embd, num_hds),
        #     Block(n_embd, num_hds),
        #     nn.LayerNorm(n_embd)
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, num_hds) for _ in range(num_layers)])
        self.lyr_nrm = nn.LayerNorm(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        # x = self.as_head(x)
        # x = self.ffn(x)
        x = self.blocks(x)
        x = self.lyr_nrm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))