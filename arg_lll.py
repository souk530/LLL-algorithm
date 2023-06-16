import numpy as np
import sys
import time
from tqdm import tqdm
def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v, b)*b for b in basis)
        #dot:内積
        basis.append(w)
    return np.array(basis)

def LLL(vectors, delta=0.75):
    THRESHOULD = 1e-10
    n = len(vectors)
    B = np.array(vectors, dtype=float)
    ortho_B = gram_schmidt(B)
    
    def mu(i, j):
        denominator = np.dot(ortho_B[j], ortho_B[j])
        if np.isclose(np.dot(ortho_B[j], ortho_B[j]), 0, atol=THRESHOULD):
            denominator = THRESHOULD
        value = np.dot(B[i], ortho_B[j]) / denominator
        return value
    
    k = 1
    while k < n:
        for j in range(k-1, -1, -1):
            mu_kj = mu(k, j)
            if abs(mu_kj) > 0.5:
                B[k] = B[k] - round(mu_kj) * B[j]
                ortho_B = gram_schmidt(B)
        
        if np.linalg.norm(ortho_B[k])**2 >= (delta - mu(k, k-1)**2) * np.linalg.norm(ortho_B[k-1])**2:
            #norm:ノルム,linalg:固有値
            k += 1
        else:
            B[k], B[k-1] = B[k-1], B[k].copy()
            ortho_B = gram_schmidt(B)
            k = max(k-1, 1)
    return B

def generate_matrix(n,m, low=1, high=10):
    matrix = np.random.randint(low, high, (n, n))
    return matrix%m

def is_linearly_independent(vectors):
    matrix = np.array(vectors)
    if np.linalg.matrix_rank(matrix) == len(vectors):
        return True
    else:
        return False
    
# テスト
num = int(sys.argv[1])
m = 37

while True:
    vectors = generate_matrix(num,m)
    print("original:\n",vectors)
    if is_linearly_independent(vectors) == False:
        print("線形従属")
        continue
    else:
        matrix_LLL = LLL(vectors)%m
        print("after LLL:\n",matrix_LLL)
        time.sleep(3)
