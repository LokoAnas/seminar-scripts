import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    
    x = np.asarray(x, dtype=np.float64)
    positive_mask = x >= 0
    negative_mask = ~positive_mask

    z = np.empty_like(x, dtype=np.float64)

    # For x >= 0: σ(x) = 1 / (1 + exp(-x))
    z[positive_mask] = 1.0 / (1.0 + np.exp(-x[positive_mask]))

    # For x < 0: σ(x) = exp(x) / (1 + exp(x))
    exp_x = np.exp(x[negative_mask])
    z[negative_mask] = exp_x / (1.0 + exp_x)

    return z


def calculate_dpo_loss(
    policy_logprob_win: float,
    ref_logprob_win: float,
    policy_logprob_lose: float,
    ref_logprob_lose: float,
    beta: float,
) -> float:
   
    if beta <= 0:
        raise ValueError("beta must be positive.")

    # Advantages relative to the reference model
    # A_win  = log π_θ(win|x) - log π_ref(win|x)
    # A_lose = log π_θ(lose|x) - log π_ref(lose|x)
    adv_win = policy_logprob_win - ref_logprob_win
    adv_lose = policy_logprob_lose - ref_logprob_lose

    # Margin encouraging the policy to favor win over lose.
    # Larger margin (A_win - A_lose) => lower loss.
    margin = beta * (adv_win - adv_lose)

    # Compute -log σ(margin)
    sigma_margin = sigmoid(margin)
    # To avoid log(0), add a tiny epsilon.
    eps = 1e-12
    loss = -np.log(sigma_margin + eps)

    return float(loss)


if __name__ == "__main__":
    # Simple sanity check using the numbers from the assignment.
    beta = 0.1
    policy_logprob_win = -1.5
    ref_logprob_win = -1.2
    policy_logprob_lose = -1.0
    ref_logprob_lose = -1.8

    loss_value = calculate_dpo_loss(
        policy_logprob_win,
        ref_logprob_win,
        policy_logprob_lose,
        ref_logprob_lose,
        beta,
    )

    print("DPO loss (example):", loss_value)
