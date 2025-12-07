import numpy as np
from Const import Const



def compute_transition_probabilities(C: Const) -> np.array:
    """
    Computes the transition probability tensor P.

    """

    K = C.K
    L = C.L
    M = C.M
    state_space = C.state_space
    input_space = C.input_space
    

    P = np.zeros((K, K, L), dtype=np.float64)


    idx = {s: i for i, s in enumerate(state_space)}


    obstacle_cache: dict[tuple, list] = {}


    Wv = np.arange(-C.V_dev, C.V_dev + 1, dtype=int)
    num_Wv = len(Wv)
    flap_distributions = {}
    for u in input_space:
        if u == C.U_strong:
            flap_distributions[u] = (
                Wv,
                np.full(num_Wv, 1.0 / num_Wv, dtype=np.float64),
            )
        else:
            flap_distributions[u] = (
                np.array([0], dtype=int),
                np.array([1.0], dtype=np.float64),
            )

    G = C.G
    gap_half = (G - 1) // 2
    X = C.X
    D_min = C.D_min
    Y_max = C.Y - 1
    V_max = C.V_max
    g = C.g
    S_h = tuple(C.S_h)
    n_h = len(S_h)
    h_default = S_h[0]

    def is_collision(y: int, d1: int, h1: int) -> bool:

        return (d1 == 0) and (abs(y - h1) > gap_half)

    def is_passing(y: int, d1: int, h1: int) -> bool:

        return (d1 == 0) and (abs(y - h1) <= gap_half)

    def spawn_probability(s_free: int) -> float:

        if s_free <= D_min - 1:
            return 0.0
        elif s_free >= X:
            return 1.0
        else:
            return (s_free - (D_min - 1)) / float(X - D_min)

    for i, state in enumerate(state_space):

        y = state[0]
        v = state[1]
        D = state[2 : 2 + M]
        H = state[2 + M : 2 + 2 * M]

        d1 = D[0]
        h1 = H[0]

        if is_collision(y, d1, h1):
            continue


        key = (y, D, H)

        if key in obstacle_cache:
            spawn_cases = obstacle_cache[key]
        else:

            hat_D = list(D)
            hat_H = list(H)

            if is_passing(y, d1, h1):

                if M == 1:
                    hat_D[0] = 0
                    hat_H[0] = h_default
                else:
                    # d_hat[0] = d2 - 1
                    hat_D[0] = D[1] - 1
                    hat_H[0] = H[1]

                    for k in range(1, M - 1):
                        hat_D[k] = D[k + 1]
                        hat_H[k] = H[k + 1]

                    hat_D[M - 1] = 0
                    hat_H[M - 1] = h_default
            else:

                hat_D[0] = D[0] - 1

            s_free = (X - 1) - sum(hat_D)
            p_spawn = spawn_probability(s_free)
            p_no_spawn = 1.0 - p_spawn


            spawn_cases = []

           
            if p_no_spawn > 0.0:
                spawn_cases.append((tuple(hat_D), tuple(hat_H), p_no_spawn))

            # Spawn branch
            if p_spawn > 0.0:
               
                m_min = None
                for j in range(1, M): 
                    if hat_D[j] == 0:
                        m_min = j
                        break
                if m_min is None:
                    m_min = M - 1

                base_prob = p_spawn / n_h

                for h_new in S_h:
                    D_new = list(hat_D)
                    H_new = list(hat_H)
                    D_new[m_min] = s_free
                    H_new[m_min] = h_new
                    spawn_cases.append((tuple(D_new), tuple(H_new), base_prob))

            obstacle_cache[key] = spawn_cases


        for l, u in enumerate(input_space):
            W_vals, W_probs = flap_distributions[u]

            for (D_new, H_new, p_s) in spawn_cases:
                if p_s == 0.0:
                    continue

                
                for w_flap, p_w in zip(W_vals, W_probs):
                    if p_w == 0.0:
                        continue

                    prob = p_s * p_w
                    if prob == 0.0:
                        continue


                    y_next = y + v
                    if y_next < 0:
                        y_next = 0
                    elif y_next > Y_max:
                        y_next = Y_max

                   
                    v_next = v + u + w_flap - g
                    if v_next < -V_max:
                        v_next = -V_max
                    elif v_next > V_max:
                        v_next = V_max


                    next_state = (y_next, v_next, *D_new, *H_new)
                    j = idx[next_state]
                    P[i, j, l] += prob


    return P
