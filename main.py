import numpy as np
import pandas as pd
import os

class HadamardMatrix:
    # (类定义保持不变，省略递归生成、提取、映射方法以保持简洁)

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
        rows = np.array(row_indices) if row_indices is not None else np.arange(R)
        cols = np.array(col_indices) if col_indices is not None else np.arange(C)
        submatrix = self.H[np.ix_(rows, cols)]
        return submatrix

    @staticmethod
    def map_to_degrees(matrix: np.ndarray) -> np.ndarray:
        """将矩阵元素 (+1, -1) 映射为角度 (0度, 180度)。"""
        degrees_matrix = np.where(matrix == 1, 0, 180)
        return degrees_matrix

    @staticmethod
    def save_to_csv(matrix: np.ndarray, filename: str):
        """将给定的矩阵保存到 CSV 文件中。"""
        df = pd.DataFrame(matrix)
        # 不保存行索引和列名，方便后续读取
        df.to_csv(filename, index=False, header=False)
        print(f"✅ 矩阵已成功保存到文件: {filename} (形状: {matrix.shape})")

    @staticmethod
    def verify_orthogonality(matrix: np.ndarray, sample_rows=2) -> np.ndarray:
        """
        验证矩阵的行向量是否两两正交，通过计算 A * A.T 实现。

        参数:
            matrix (np.ndarray): M x N 的矩阵。
            sample_rows (int): 打印内积矩阵中的前 n 行以供示例验证。

        返回:
            np.ndarray: 行内积矩阵 (M x M)。
        """
        print(f"\n5. 验证 {matrix.shape[0]}x{matrix.shape[1]} 矩阵的行正交性...")

        # 计算行内积矩阵：A @ A.T
        inner_product_matrix = np.dot(matrix, matrix.T)

        M, N = matrix.shape

        print(f"   -> 矩阵 A @ A.T 的形状: {inner_product_matrix.shape}")

        # 验证对角线元素 (行向量长度的平方)
        diagonal_values = np.diag(inner_product_matrix)
        is_diagonal_correct = np.all(diagonal_values == N)
        print(f"   -> 对角线元素 (长度平方) 是否都等于 N={N}: {is_diagonal_correct}")

        # 验证非对角线元素 (正交性)
        # 创建一个对角矩阵 D，其对角线值与 inner_product_matrix 相同
        D = np.diag(diagonal_values)
        # 检查 A @ A.T - D 是否只包含 0
        off_diagonal_sum = np.sum(np.abs(inner_product_matrix - D))

        if np.isclose(off_diagonal_sum, 0):
            print("   -> 结论: **矩阵行向量是两两正交的**。")
        else:
            print("   -> 结论: **矩阵行向量不再是两两正交的** (非对角线元素非零)。")

        print(f"   -> 打印前 {sample_rows}x{sample_rows} 行内积矩阵 (A @ A.T) 部分:")
        # 打印部分内积矩阵，以便用户手动观察
        print(inner_product_matrix[:sample_rows, :sample_rows])

        return inner_product_matrix

# -----------------------------------------------------------
# 主要实现函数 (修改后)
# -----------------------------------------------------------

def generate_and_process_hadamard_data(M: int, N: int):
    """
    根据 M 和 N 的要求生成 Hadamard 矩阵，提取子集并进行正交性验证和保存。
    """
    if M < N:
        raise ValueError("参数错误: M 必须大于或等于 N。")
    if M & (M - 1) != 0 or M < 1:
        raise ValueError("参数错误: M 必须是 2 的幂次方 (1, 2, 4, 8, ...)")

    print(f"\n=======================================================")
    print(f"--- 开始处理 M={M}, N={N} 的 Hadamard 矩阵 ---")

    try:
        # 1. 生成 M x M 的 Hadamard 矩阵
        print(f"1. 生成 {M}x{M} Hadamard 矩阵...")
        had_tool = HadamardMatrix(M)

        # 2. 抽取 M x N 矩阵 (选择所有 M 行，前 N 列)
        print(f"2. 抽取 M x N ({M}x{N}) 子矩阵...")
        H_M_N = had_tool.extract_submatrix(R=M, C=N)

        # 3. 保存 M x N 矩阵到 ZJ.csv
        HadamardMatrix.save_to_csv(H_M_N, "ZJ.csv")

        # 4. 将 M x N 矩阵映射为角度并保存
        H_M_N_DEG = HadamardMatrix.map_to_degrees(H_M_N)
        HadamardMatrix.save_to_csv(H_M_N_DEG, "ZJMXN.csv")

        # 5. 验证 M x N 矩阵的正交性
        HadamardMatrix.verify_orthogonality(H_M_N)

        print("\n--- 所有操作完成 ---")

    except ValueError as e:
        print(f"处理失败: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 示例调用 ---
if __name__ == '__main__':

    # 示例 A: M=16, N=8 (N < M，预期：不正交)
    generate_and_process_hadamard_data(M=16, N=8)

    # 示例 B: M=4, N=4 (N = M，预期：完全正交)
    generate_and_process_hadamard_data(M=4, N=4)
