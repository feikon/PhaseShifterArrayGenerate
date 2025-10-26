import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class HadamardMatrix:
    """
    用于生成 Hadamard 矩阵、提取子集、角度映射的工具类。
    阶数 N 必须是 2 的幂次方。
    """
    def __init__(self, N: int):
        self.N = N
        self.H = self._recursive_hadamard(N)

    def _recursive_hadamard(self, n: int) -> np.ndarray:
        """使用 Sylvester 构造法递归生成 n 阶 Hadamard 矩阵。"""
        if n < 1 or (n & (n - 1) != 0):
            raise ValueError(f"阶数 N={n} 必须是 2 的幂次方 (1, 2, 4, 8, ...)")

        if n == 1:
            return np.array([[1]], dtype=np.int8)

        H_half = self._recursive_hadamard(n // 2)
        H_n = np.block([[H_half, H_half], [H_half, -H_half]])
        return H_n.astype(np.int8)

    def get_matrix(self) -> np.ndarray:
        """返回生成的 Hadamard 矩阵 (元素为 +1, -1)。"""
        return self.H

    def extract_submatrix(self, R: int, C: int, row_indices=None, col_indices=None) -> np.ndarray:
        """从 N x N 的 Hadamard 矩阵中提取 R x C 的子矩阵 (元素为 +1, -1)。"""
        if R > self.N or C > self.N:
            raise ValueError(f"目标维度 {R}x{C} 不能大于原始 Hadamard 矩阵的阶数 {self.N}")

        # 确定索引 (M*N 矩阵，因此 R=M, C=N)
        rows = np.array(row_indices) if row_indices is not None else np.arange(R)
        # 默认选择前 N 列
        cols = np.array(col_indices) if col_indices is not None else np.arange(C)

        submatrix = self.H[np.ix_(rows, cols)]
        return submatrix

    @staticmethod
    def map_to_degrees(matrix: np.ndarray) -> np.ndarray:
        """
        将矩阵元素 (+1, -1) 映射为角度 (0度, 180度)。
        1 -> 0°
        -1 -> 180°
        """
        # 使用 NumPy 数组操作进行映射
        degrees_matrix = np.where(matrix == 1, 0, 180)
        return degrees_matrix

    @staticmethod
    def save_to_csv(matrix: np.ndarray, filename: str):
        """将给定的矩阵保存到 CSV 文件中。"""
        df = pd.DataFrame(matrix)
        df.to_csv(filename, index=False, header=False) # 不保存行索引和列名
        print(f"✅ 矩阵已成功保存到文件: {filename} (形状: {matrix.shape})")

# -----------------------------------------------------------
# 主要实现函数
# -----------------------------------------------------------

def generate_and_process_hadamard_data(M: int, N: int):
    """
    根据 M 和 N 的要求生成 Hadamard 矩阵，提取子集并保存结果。

    参数:
        M (int): 原始 Hadamard 矩阵的阶数 M x M。M 必须是 2 的幂次方。
        N (int): 提取子矩阵的列数 M x N。M >= N。
    """
    if M < N:
        raise ValueError("参数错误: M 必须大于或等于 N。")
    if M & (M - 1) != 0 or M < 1:
        raise ValueError("参数错误: M 必须是 2 的幂次方 (1, 2, 4, 8, ...)")

    print(f"--- 开始处理 M={M}, N={N} 的 Hadamard 矩阵 ---")

    try:
        # 1. 生成 M x M 的 Hadamard 矩阵
        print(f"1. 生成 {M}x{M} Hadamard 矩阵...")
        had_tool = HadamardMatrix(M)
        H_M = had_tool.get_matrix()

        # 2. 抽取 M x N 矩阵 (选择所有 M 行，前 N 列)
        print(f"2. 抽取 M x N ({M}x{N}) 子矩阵...")
        # 提取 M 行（所有行），N 列（前 N 列）
        H_M_N = had_tool.extract_submatrix(R=M, C=N)

        # 3. 保存 M x N 矩阵到 ZJ.csv
        ZJ_FILENAME = "ZJ.csv"
        HadamardMatrix.save_to_csv(H_M_N, ZJ_FILENAME)

        # 4. 将 M x N 矩阵映射为角度
        print("4. 将 M x N 矩阵映射为角度 (deg)...")
        H_M_N_DEG = HadamardMatrix.map_to_degrees(H_M_N)

        # 5. 保存角度矩阵到 ZJMXN.csv
        ZJMXN_FILENAME = f"ZJ{M}X{N}.csv"
        HadamardMatrix.save_to_csv(H_M_N_DEG, ZJMXN_FILENAME)

        print("\n--- 所有操作完成 ---")

    except ValueError as e:
        print(f"处理失败: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 示例调用 ---
if __name__ == '__main__':
    # 示例 1: M=16, N=8
    M_val_1 = 16
    N_val_1 = 8
    generate_and_process_hadamard_data(M_val_1, N_val_1)

    # 示例 2: M=4, N=4 (抽取完整的 Hadamard 矩阵)
    # M_val_2 = 4
    # N_val_2 = 4
    # generate_and_process_hadamard_data(M_val_2, N_val_2)

    # 示例：验证保存的 CSV 文件内容（可选）
    # print("\n--- 验证 CSV 文件内容 ---")
    # print(pd.read_csv("ZJ.csv", header=None).head())
    # print(pd.read_csv("ZJMXN.csv", header=None).head())