import numpy as np

def wmmse_beamformer(H, P, N0, max_iter=100, epsilon=1e-5):
    M, K = H.shape  # 天线数和用户数

    # 初始化beamformer
    W = np.random.randn(M, K) + 1j * np.random.randn(M, K)
    W = W / np.linalg.norm(W, axis=0)

    for iter in range(max_iter):
        # 计算干扰加噪声协方差矩阵
        Q = np.zeros((K, K), dtype=complex)
        for k in range(K):
            Q[k, k] = N0 + np.sum(np.abs(H[:, k].conj().T @ W[:, np.arange(K) != k]) ** 2)
        
        # 更新beamformer
        W_new = np.zeros((M, K), dtype=complex)
        for k in range(K):
            Hk = H[:, k].reshape(-1, 1)
            Wk = W[:, k].reshape(-1, 1)
            W_new[:, k] = Hk * (P[k] * Wk.conj().T @ Hk) / (Hk.conj().T @ Q @ Hk + epsilon)
        W_new = W_new / np.linalg.norm(W_new, axis=0)
        
        # 判断是否收敛
        if np.linalg.norm(W_new - W) < epsilon:
            break
        
        W = W_new
    
    return W

# 示例用法
M = 4  # 天线数
K = 3  # 用户数
H = np.random.randn(M, K) + 1j * np.random.randn(M, K)  # 信道矩阵
P = np.ones(M)  # 用户功率
N0 = 1  # 噪声功率

W = wmmse_beamformer(H, P, N0)
print("Beamformer:")
print(W)