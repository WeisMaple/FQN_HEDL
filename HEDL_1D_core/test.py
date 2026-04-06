import numpy as np
# X: (N, C, L)  例如 1000 个样本, 1 通道, 序列长 500
# y: (N,)  整数标签 0~4
np.savez(r'F:\QML\Papers\CODES\FQN_HEDL\Data\ts_demo.npz', X=X, y=y)