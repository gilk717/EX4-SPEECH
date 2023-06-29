import numpy as np
import sys


def compute_alpha(y, target_word, tokens):
    blank_token = 0
    # init z
    z = [blank_token for _ in range(len(target_word) * 2 + 1)]
    for i in range(len(z)):
        if i % 2 == 1:
            z[i] = tokens.index(target_word[int(i / 2)]) + 1
    S = len(z)
    T = y.shape[0]
    # init alpha
    alpha = np.zeros((S, T))
    alpha[0, 0] = y[0, blank_token]
    alpha[1, 0] = y[0, z[1]]

    for t in range(1, T):
        for s in range(0, S):
            if s == 0:
                alpha[s, t] = (alpha[s, t - 1]) * y[t, blank_token]
            if s == 1:
                alpha[s, t] = (alpha[s, t - 1] + alpha[s - 1, t - 1]) * y[t, z[s]]
            else:
                if z[s] == blank_token or z[s] == z[s - 2]:
                    alpha[s, t] = (alpha[s - 1, t - 1] + alpha[s, t - 1]) * y[t, z[s]]
                else:
                    alpha[s, t] = (alpha[s - 2, t - 1] + alpha[s - 1, t - 1] + alpha[s, t - 1]) * y[t, z[s]]

    return alpha


def print_p(p: float):
    print("%.3f" % p)


def get_p(alpha):
    return alpha[-1, -1] + alpha[-2, -1]


if __name__ == "__main__":
    path, word, available_tokens = sys.argv[1], sys.argv[2], sys.argv[3]
    mat_y = np.load(path)
    res_alpha = compute_alpha(mat_y, word, available_tokens)
    prob = get_p(res_alpha)
    print_p(prob)
