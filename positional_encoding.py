import numpy as np
import matplotlib.pyplot as plt

def create_positional_encoding(max_len: int, d_model: int) -> np.ndarray:

    # Create position indices (max_len x 1)
    positions = np.arange(max_len)[:, np.newaxis]  # shape (max_len, 1)
    # Create dimension indices (1 x d_model)
    dims = np.arange(d_model)[np.newaxis, :]       # shape (1, d_model)

    # Compute the angle rates: 1 / (10000^(2i/d_model))
    angle_rates = 1 / (10000 ** ((2 * (dims // 2)) / np.float32(d_model)))

    # Compute the angle radians (pos * angle_rates)
    angle_rads = positions * angle_rates  # shape (max_len, d_model)

    # Apply sin to even indices; cos to odd indices
    pos_encoding = np.zeros_like(angle_rads)
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])   # even indices
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])   # odd indices

    return pos_encoding.astype(np.float32)


def apply_encoding(embeddings: np.ndarray, pos_encodings: np.ndarray) -> np.ndarray:

    seq_len, d_model = embeddings.shape
    # Ensure pos_encodings has at least seq_len rows
    if pos_encodings.shape[0] < seq_len:
        raise ValueError("pos_encodings must have at least seq_len rows")
    return embeddings + pos_encodings[:seq_len, :]


# -------------------------
# Demonstration (5x8 example)
# -------------------------
if __name__ == "__main__":
    np.random.seed(42)

    seq_len = 5
    d_model = 8

    # Hypothetical 5-word sentence embeddings (random for demonstration)
    embeddings = np.random.randn(seq_len, d_model).astype(np.float32)

    # Create positional encodings and apply
    pos_enc = create_positional_encoding(max_len=50, d_model=d_model)
    encoded = apply_encoding(embeddings, pos_enc)

    # Print shapes and first rows for quick inspection
    print("Embeddings shape:", embeddings.shape)
    print("Positional encodings shape:", pos_enc.shape)
    print("Encoded shape:", encoded.shape)
    print("\nFirst row of embeddings:\n", embeddings[0])
    print("\nFirst row of positional encoding:\n", pos_enc[0])
    print("\nFirst row after addition:\n", encoded[0])

    # Visualize positional encodings for the first 5 positions (heatmap)
    plt.figure(figsize=(6, 3))
    plt.title("Positional Encodings (first 5 positions, d_model=8)")
    plt.imshow(pos_enc[:seq_len, :], aspect='auto')
    plt.xlabel("Embedding Dimension (0..7)")
    plt.ylabel("Position (0..4)")
    plt.colorbar(label="value")
    plt.tight_layout()
    plt.show()
