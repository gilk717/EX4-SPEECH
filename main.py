import numpy as np
import sys


def print_p(p: float):
    print("%.3f" % p)


def get_prob(alpha, y, blank_token):
    T, _ = y.shape
    return alpha[-1, T - 1] + alpha[-2, T - 1]


def compute_ctc_alpha(y, z, tokens):
    blank_token = len(tokens)
    tokens = tokens + 'ϵ'
    z = [blank_token] + [tokens.index(token) for token in z] + [blank_token]
    S, T = len(z), y.shape[0]
    alpha = np.zeros((S, T))

    # Initialize first column
    alpha[0, 0], alpha[1, 0] = y[0, z[0]], y[0, z[1]]
    for s in range(2, S):
        alpha[s, 0] = 0

    for t in range(1, T):
        for s in range(1, S):  # Start from s=1
            if s == 1:
                alpha[s, t] = (alpha[s, t - 1] + alpha[s - 1, t - 1]) * y[t, z[s]]
            else:
                if z[s] == blank_token or z[s] == z[s - 2]:
                    alpha[s, t] = (alpha[s, t - 1] + alpha[s - 1, t - 1]) * y[t, z[s]]
                else:
                    alpha[s, t] = (alpha[s, t - 1] + alpha[s - 1, t - 1] + alpha[s - 2, t - 1]) * y[t, z[s]]

    return alpha


def main():
    # Parse command line arguments
    path, target_sequence, tokens = sys.argv[1], sys.argv[2], sys.argv[3]

    # Load network outputs
    y = np.load(path)

    # Check if all characters in target_sequence are in tokens
    for char in target_sequence:
        if char not in tokens:
            raise ValueError(f"Character {char} is not in tokens")

    # Calculate alpha matrix
    alpha = compute_ctc_alpha(y, target_sequence, tokens)

    # Calculate probability
    prob = get_prob(alpha, y, len(tokens))

    # Print probability with the specified format
    print_p(prob)


if __name__ == "__main__":
    main()