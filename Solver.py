#import time
import numpy as np
from scipy.sparse import csr_matrix, eye,vstack
from scipy.sparse.linalg import spsolve
from Const import Const

from scipy.sparse import csr_matrix





def solution(C: Const) -> tuple[np.array, np.array]:
    """
    Solves the problem 
    """
    
    

    #solver_start_time = time.time()
# %% mystate space + caching alreafy efficient = 0.00123 s 
    
    
    
    S_y = np.array(C.S_y, dtype=np.int32)
    S_v = np.array(C.S_v, dtype=np.int32)
    S_d1 = np.array(C.S_d1, dtype=np.int32)
    S_d = np.array(C.S_d, dtype=np.int32)
    S_h = np.array(C.S_h, dtype=np.int32)

    M = C.M
    
   
    axes = [S_y, S_v, S_d1] + [S_d]*(M-1) + [S_h]*M
    grids = np.meshgrid(*axes, indexing='ij')
    full_space = np.stack(grids, axis=-1).reshape(-1, len(axes))


    col_d1 = full_space[:, 2]
    cols_d_rest = full_space[:, 3 : 2 + M]
    cols_h_rest = full_space[:, 2 + M + 1 :]
    

    mask = (np.sum(full_space[:, 2:2+M], axis=1) <= C.X - 1)


    if M > 1:
        d2 = cols_d_rest[:, 0]

        mask &= ~((d2 == 0) & (col_d1 <= 0))
        

        invalid_dh = (cols_d_rest == 0) & (cols_h_rest != S_h[0])
        mask &= ~np.any(invalid_dh, axis=1)


        if cols_d_rest.shape[1] >= 2:
            zero_gap = (cols_d_rest[:, :-1] == 0) & (cols_d_rest[:, 1:] != 0)
            mask &= ~np.any(zero_gap, axis=1)

    states = full_space[mask]
    

    X = C.X
    Y_max = C.Y - 1
    V_max = C.V_max
    D_min = C.D_min
    gap_half = (C.G - 1) // 2 
    
    n_h = len(S_h)
    input_space = [C.U_no_flap, C.U_weak, C.U_strong]
    
    K = len(states)
    L = len(input_space)
    V_dev = C.V_dev
    
    #print(f"set upped initial cache in {time.time()-solver_start_time}")

   
# %% get Q optmized currently 1e-5 s 
   

    base_cost = -1.0
        
    cost_map = {
            C.U_no_flap: base_cost,
            C.U_weak: base_cost + C.lam_weak,
            C.U_strong: base_cost + C.lam_strong
        }
    cost_vec = np.array([cost_map[u] for u in input_space])
        
        
    Q = np.tile(cost_vec, (K, 1))
        
    
    
# %% get P  #to improve currently 0.01 s 
    # 2. Get Sparse P
    
    #time_p = time.time()
    if True:
        
        
        state_dtype = states.dtype
        
        raw_view_dtype = np.dtype((np.void, states.shape[1] * state_dtype.itemsize))
        states_contiguous = np.ascontiguousarray(states)
        states_view = states_contiguous.view(raw_view_dtype).ravel()
        sort_idx = np.argsort(states_view)
        sorted_states_view = states_view[sort_idx]

        
        curr_y = states[:, 0]
        curr_v = states[:, 1]
        curr_D = states[:, 2 : 2+M]
        curr_H = states[:, 2+M : 2+2*M]
        d1 = curr_D[:, 0]
        h1 = curr_H[:, 0]

        passing_mask = (d1 == 0) & (np.abs(curr_y - h1) <= gap_half)
        valid_mask = ~((d1 == 0) & (~passing_mask))
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            P_sparse = [csr_matrix((K, K)) for _ in range(L)]
            

        v_y = curr_y[valid_indices]
        v_v = curr_v[valid_indices]
        v_D = curr_D[valid_indices]
        v_H = curr_H[valid_indices]
        v_passing = passing_mask[valid_indices]

        hat_D = v_D.copy()
        hat_H = v_H.copy()
        hat_D[~v_passing, 0] -= 1 
        
        if M == 1:
            hat_D[v_passing, 0] = 0
            hat_H[v_passing, 0] = S_h[0]
        else:
            hat_D[v_passing, 0:M-1] = v_D[v_passing, 1:M]
            hat_D[v_passing, 0] -= 1 
            hat_D[v_passing, M-1] = 0
            hat_H[v_passing, 0:M-1] = v_H[v_passing, 1:M]
            hat_H[v_passing, M-1] = S_h[0]

        total_span = np.sum(hat_D, axis=1)
        s_free = (X - 1) - total_span
        p_spawn = np.clip((s_free - (D_min - 1)) / float(X - D_min), 0.0, 1.0)
        p_no_spawn = 1.0 - p_spawn
        mask_spawn = (p_spawn > 0)
        mask_no_spawn = (p_no_spawn > 0)

        d_new = np.clip(s_free, D_min, X - 1).astype(np.int32)
        spawn_D = hat_D.copy()
        
        target_indices = np.zeros(len(spawn_D), dtype=int)
        if M > 1:
            search_cols = spawn_D[:, 1:]
            is_zero = (search_cols == 0)
            m_min_idx = np.argmax(is_zero, axis=1)
            found_zero = np.any(is_zero, axis=1)
            target_indices = np.where(found_zero, m_min_idx + 1, M - 1)
            row_arange = np.arange(len(spawn_D))
            spawn_D[row_arange, target_indices] = d_new
        else:
            spawn_D[:, 0] = d_new

        P_sparse = []
        original_indices = valid_indices

        for u_val in input_space:
            if u_val == C.U_strong:
                w_vals = np.arange(-V_dev, V_dev + 1)
                w_probs = np.full(len(w_vals), 1.0/len(w_vals))
            else:
                w_vals = np.array([0])
                w_probs = np.array([1.0])

            all_rows, all_cols, all_data = [], [], []

            for w, w_prob in zip(w_vals, w_probs):
                if w_prob == 0: continue
                v_next = np.clip(v_v + u_val + w - C.g, -V_max, V_max)
                y_next = np.clip(v_y + v_v, 0, Y_max)

                idx_ns = np.where(mask_no_spawn)[0]
                if len(idx_ns) > 0:
                    cand = np.hstack([y_next[idx_ns, None], v_next[idx_ns, None], hat_D[idx_ns], hat_H[idx_ns]]).astype(np.int32)
                    cand_view = np.ascontiguousarray(cand).view(raw_view_dtype).ravel()
                    found_pos = np.searchsorted(sorted_states_view, cand_view)
                    found_pos = np.clip(found_pos, 0, len(sorted_states_view)-1)
                    is_match = (sorted_states_view[found_pos] == cand_view)
                    match_indices = np.where(is_match)[0]
                    if len(match_indices) > 0:
                        all_rows.append(original_indices[idx_ns[match_indices]])
                        all_cols.append(sort_idx[found_pos[match_indices]])
                        all_data.append(p_no_spawn[idx_ns[match_indices]] * w_prob)

                idx_s = np.where(mask_spawn)[0]
                if len(idx_s) > 0:
                    base_prob_vals = (p_spawn[idx_s] / n_h) * w_prob
                    t_idx = target_indices[idx_s]
                    for h_new_val in S_h:
                        c_H = hat_H[idx_s].copy()
                        c_H[np.arange(len(c_H)), t_idx] = h_new_val
                        cand = np.hstack([y_next[idx_s, None], v_next[idx_s, None], spawn_D[idx_s], c_H]).astype(np.int32)
                        cand_view = np.ascontiguousarray(cand).view(raw_view_dtype).ravel()
                        found_pos = np.searchsorted(sorted_states_view, cand_view)
                        found_pos = np.clip(found_pos, 0, len(sorted_states_view)-1)
                        is_match = (sorted_states_view[found_pos] == cand_view)
                        match_indices = np.where(is_match)[0]
                        if len(match_indices) > 0:
                            all_rows.append(original_indices[idx_s[match_indices]])
                            all_cols.append(sort_idx[found_pos[match_indices]])
                            all_data.append(base_prob_vals[match_indices])

            if all_rows:
                mat = csr_matrix((np.concatenate(all_data), (np.concatenate(all_rows), np.concatenate(all_cols))), shape=(K, K))
                #mat.sum_duplicates()
                P_sparse.append(mat)
            else:
                P_sparse.append(csr_matrix((K, K)))

  
    
    #print(f"got P in {time.time()-time_p}")    
    #print(f"got P and Q in {time.time()-solver_start_time}") #modified 
    




# %% POlicy iteration improvable implementing efficiently VI currently 0.06s ideally if we can get the effective iterations down to 4-6 we done 

    
    def solve_hybrid_turbo(P_sparse, Q, gamma=1.0,ratio_delta_vi= 0.65, vi_iters_max=2000, pi_max_iter=500, tolerance=1e-8, verbose=False): #FASTER IN VI


        #solver_start = time.time()
        K, L = Q.shape
        

        #begin = time.time()
        if verbose:
            print(f"\nwarm up with {vi_iters_max} iterations")


        P_stack = vstack(P_sparse, format='csr')


        Q_flat = Q.T.reshape(-1) 

        J = np.zeros(K, dtype=np.float64)
        

        for i in range(vi_iters_max):

            J_old =J.copy()
            J_next_all = Q_flat + gamma * P_stack.dot(J)
            

            J = np.min(J_next_all.reshape(L, K), axis=0)
            delta = np.max(np.abs(J - J_old))
            if (delta < ratio_delta_vi):
                break


        J_next_all = Q_flat + gamma * P_stack.dot(J)
        current_policy = np.argmin(J_next_all.reshape(L, K), axis=0)
        
            

        if verbose:
            #print(f"turbo VI time {time.time()-begin}")
            print(f"Phase 1 Complete. Mean Cost: {np.mean(J):.4f}")
            print(f"Phase 2: Switching to Policy Iteration")
            print(f"{'Iter':<5} , {'Delta (J)':<12} ,{'Pol. Chg':<10} , {'Step (s)':<12}")
            


        I = eye(K, format='csr')
        iter_times = []
        converged = False
        
        for i in range(pi_max_iter):
            #iter_start = time.time()
            J_old = J.copy()


            P_pi_parts = []
            for l in range(L):
                row_mask = (current_policy == l)
                if not np.any(row_mask): continue
                P_pi_parts.append(P_sparse[l].multiply(row_mask[:, None]))
                
            P_pi = sum(P_pi_parts)
            Q_pi = Q[np.arange(K), current_policy]

            try:

                if gamma == 1.0:
                    J = spsolve(I - P_pi, Q_pi)
                else:
                    J = spsolve(I - gamma * P_pi, Q_pi)
            except RuntimeError:
                if verbose: print(f"Singular matrix, instead we use iterative method.")
                for _ in range(1000):
                    J = Q_pi + gamma * P_pi.dot(J)


            J_next_all = Q_flat + gamma * P_stack.dot(J)
            new_policy = np.argmin(J_next_all.reshape(L, K), axis=0)


            delta = np.max(np.abs(J - J_old))
            policy_changes = np.count_nonzero(current_policy != new_policy)
            #iter_dt = time.time() - iter_start
            #iter_times.append(iter_dt)

            if verbose:
                print(f"{i:<5} , {delta:.2e}     ,{policy_changes:<10} ,iter_dt:.4f")

            if policy_changes == 0 or delta < tolerance:
                converged = True
                if verbose: print(f"-> Converged.")
                break
            
            current_policy = new_policy


        return J,current_policy,converged,0#time.time()-solver_start




# %% mapp to indexes, compute opt_policy and return  #fully optimal no need for further optimisation 1e-5 s 
    
    
    
    
    J,new_policy,converged,running_time  = solve_hybrid_turbo(P_sparse, Q, gamma=1.0, ratio_delta_vi=0.65,vi_iters_max=500, pi_max_iter=500, tolerance=1e-8, verbose=False)
    #best for now vi_iters = 312
    #print(f"\ntotal running time inside solver turbo : {running_time}\n")
    
    

    opt_policy = np.array(input_space)[new_policy].astype(np.int32)

    
    #print(f"\ntotal running time inside solution : {time.time() - solver_start_time}\n")
    

# %% returning   
    return J, opt_policy

