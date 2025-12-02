import numpy as np
import matplotlib.pyplot as plt

# Import the positional encoding helper (assumes both files are in same directory)
from positional_encoding import create_positional_encoding, apply_encoding

def qkv_projection(embeddings: np.ndarray, d_k: int = None, seed: int = None):

    if seed is not None:
        np.random.seed(seed)

    seq_len, d_model = embeddings.shape
    if d_k is None:
        d_k = d_model

    # Simulate weight matrices Wq, Wk, Wv (d_model x d_k)
    Wq = np.random.randn(d_model, d_k).astype(np.float32) * (1.0 / np.sqrt(d_model))
    Wk = np.random.randn(d_model, d_k).astype(np.float32) * (1.0 / np.sqrt(d_model))
    Wv = np.random.randn(d_model, d_k).astype(np.float32) * (1.0 / np.sqrt(d_model))

    # Project embeddings: shape (seq_len, d_k)
    Q = embeddings @ Wq
    K = embeddings @ Wk
    V = embeddings @ Wv

    return Q, K, V

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax along specified axis.
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray):

    d_k = Q.shape[-1]
    # Compute raw scores: (seq_len, seq_len)
    scores = Q @ K.T  # shape: (seq_len, seq_len)

    # Scale
    scores = scores / np.sqrt(d_k)

    # Softmax to get attention weights (for each query row)
    attention_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    output = attention_weights @ V  # shape: (seq_len, d_k)

    return output, attention_weights

# -------------------------
# Demonstration (main)
# -------------------------
if __name__ == "__main__":
    # Recreate the same 5x8 embeddings as the positional_encoding demo
    np.random.seed(42)
    seq_len = 5
    d_model = 8

    embeddings = np.random.randn(seq_len, d_model).astype(np.float32)

    # Apply positional encodings (same process used in specialist demo)
    pos_enc = create_positional_encoding(max_len=50, d_model=d_model)
    encoded = apply_encoding(embeddings, pos_enc)

    print("Using embeddings (with positional encodings) of shape:", encoded.shape)

    # Project into Q, K, V (single-head)
    Q, K, V = qkv_projection(encoded, d_k=d_model, seed=1234)

    # Run scaled dot-product attention
    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    # Print results
    np.set_printoptions(precision=4, suppress=True)
    print("\nAttention weights (shape {}):\n".format(attn_weights.shape), attn_weights)
    print("\nContext vectors (output) shape {}:\n".format(output.shape), output)

    # Visualize attention weights
    plt.figure(figsize=(5, 4))
    plt.title("Scaled Dot-Product Attention Weights\n(seq_len x seq_len)")
    plt.imshow(attn_weights, aspect='auto')
    plt.xlabel("Key position (0..4)")
    plt.ylabel("Query position (0..4)")
    plt.colorbar(label="attention weight")
    plt.tight_layout()
    plt.show()
