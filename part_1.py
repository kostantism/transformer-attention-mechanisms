import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


E = np.load('E.npy')
W_K_cross = np.load('W_K_cross.npy')
W_K_self = np.load('W_K_self.npy')
W_Q_cross = np.load('W_Q_cross.npy')
W_Q_self = np.load('W_Q_self.npy')
W_V_cross = np.load('W_V_cross.npy')
W_V_self = np.load('W_V_self.npy')

print("E:", E.shape)
print("W_Q_self:", W_Q_self.shape)
print("W_K_self:", W_K_self.shape)
print("W_V_self:", W_V_self.shape)
print("W_Q_cross:", W_Q_cross.shape)
print("W_K_cross:", W_K_cross.shape)
print("W_V_cross:", W_V_cross.shape)
print("\n")

def scaled_dot_product_attention(Q, K, V):
    #find dk from K
    d_k = K.shape[-1]

    #calculate attention scores
    attn_scores = (Q @ K.T) / np.sqrt(d_k)

    #softmax is applied to obtain attention probabilities
    attn_probs = np.exp(attn_scores) / np.sum(np.exp(attn_scores), axis=-1, keepdims=True)

    #multiply by values to obtain the final output
    output = attn_probs @ V

    return output, attn_probs



X_english = E[[0, 1, 2]]
X_greek = E[[3, 4, 5, 6, 7]]

Q_self = X_english @ W_Q_self
K_self = X_english @ W_K_self
V_self = X_english @ W_V_self

Y_self, A_self = scaled_dot_product_attention(Q_self, K_self, V_self)

print("Self-Attention")
print("Shape of Q_self:", Q_self.shape)
print("Shape of K_self:", K_self.shape)
print("Shape of V_self:", V_self.shape)
print("Shape of A_self:", A_self.shape)
print("Shape of Y_self:", Y_self.shape)
print("\n")

Q_cross = X_greek @ W_Q_cross
K_cross = X_english @ W_K_cross
V_cross = X_english @ W_V_cross

Y_cross, A_cross = scaled_dot_product_attention(Q_cross, K_cross, V_cross)

print("Cross-Attention")
print("Shape of Q_cross:", Q_cross.shape)
print("Shape of K_cross:", K_cross.shape)
print("Shape of V_cross:", V_cross.shape)
print("Shape of A_cross:", A_cross.shape)
print("Shape of Y_cross:", Y_cross.shape)
print("\n")

Q_self_t = torch.tensor(Q_self, dtype=torch.float32)
K_self_t = torch.tensor(K_self, dtype=torch.float32)
V_self_t = torch.tensor(V_self, dtype=torch.float32)

Y_self_torch = F.scaled_dot_product_attention(Q_self_t, K_self_t, V_self_t, scale=1.0 / np.sqrt(W_K_self.shape[-1]))

self_compare = np.allclose(Y_self, Y_self_torch, atol=1e-5)

print("Self-Attention compared to Pytorch:", self_compare)
print("\n")

Q_cross_t = torch.tensor(Q_cross, dtype=torch.float32)
K_cross_t = torch.tensor(K_cross, dtype=torch.float32)
V_cross_t = torch.tensor(V_cross, dtype=torch.float32)

Y_cross_torch = F.scaled_dot_product_attention(Q_cross_t, K_cross_t, V_cross_t, scale=1.0 / np.sqrt(W_K_cross.shape[-1]))

cross_compare = np.allclose(Y_cross, Y_cross_torch, atol=1e-5)

print("Cross-Attention compared to Pytorch:", cross_compare)
print("\n")


words_english = ["Maria", "isn't", "here"]
words_greek = ["Δεν", "είναι", "εδώ", "η", "Μαρία"]

plt.close("all")

plt.figure()
plt.title("Self-Attention (english)")
plt.xlabel("keys")
plt.ylabel("queries")
plt.imshow(A_self, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
plt.colorbar(label='Attention Weight')
plt.xticks(range(len(words_english)), words_english)
plt.yticks(range(len(words_english)), words_english)
plt.tight_layout()
plt.show()

plt.figure()
plt.title("Cross-Attention (greek to english)")
plt.xlabel("keys")
plt.ylabel("queries")
plt.imshow(A_cross, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0)
plt.colorbar(label='Attention Weight')
plt.xticks(range(len(words_english)), words_english)
plt.yticks(range(len(words_greek)), words_greek)
plt.tight_layout()
plt.show()