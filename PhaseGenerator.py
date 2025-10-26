import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
from typing import Tuple

# --- 核心 Hadamard 矩阵逻辑类 (保持不变) ---

class HadamardMatrix:
    """
    用于生成 Hadamard 矩阵、提取子集、角度映射的工具类。
    """
    def __init__(self, N: int):
        self.N = N
        self.H = self._recursive_hadamard(N)

    def _recursive_hadamard(self, n: int) -> np.ndarray:
        """使用 Sylvester 构造法递归生成 n 阶 Hadamard 矩阵。"""
        if n < 1 or (n & (n - 1) != 0):
            raise ValueError(f"阶数 M={n} 必须是 2 的幂次方。")
        if n == 1:
            return np.array([[1]], dtype=np.int8)
        H_half = self._recursive_hadamard(n // 2)
        H_n = np.block([[H_half, H_half], [H_half, -H_half]])
        return H_n.astype(np.int8)

    def get_matrix(self) -> np.ndarray:
        """返回生成的 Hadamard 矩阵 (元素为 +1, -1)。"""
        return self.H

    def extract_submatrix(self, R: int, C: int) -> np.ndarray:
        """从 N x N 的 Hadamard 矩阵中提取 R x C 的子矩阵 (元素为 +1, -1)。"""
        if R > self.N or C > self.N:
            raise ValueError(f"目标维度 {R}x{C} 不能大于原始 Hadamard 矩阵的阶数 {self.N}")

        # 提取前 R 行，前 C 列
        rows = np.arange(R)
        cols = np.arange(C)
        submatrix = self.H[np.ix_(rows, cols)]
        return submatrix

    @staticmethod
    def map_to_degrees(matrix: np.ndarray) -> np.ndarray:
        """将矩阵元素 (+1, -1) 映射为角度 (0度, 180度)。"""
        return np.where(matrix == 1, 0, 180)

    @staticmethod
    def save_to_csv(matrix: np.ndarray, filename: str):
        """将给定的矩阵保存到 CSV 文件中。"""
        df = pd.DataFrame(matrix)
        df.to_csv(filename, index=False, header=False)

    @staticmethod
    def verify_orthogonality(matrix: np.ndarray) -> Tuple[bool, float]:
        """验证矩阵的行向量是否两两正交。"""
        # 计算行内积矩阵：A @ A.T
        inner_product_matrix = np.dot(matrix, matrix.T)
        M_dim = matrix.shape[1] # 向量的维度 (即 M)

        diagonal_values = np.diag(inner_product_matrix)
        D = np.diag(diagonal_values)
        off_diagonal_sum = np.sum(np.abs(inner_product_matrix - D))

        # 只有当所有非对角线内积都接近 0，且对角线内积等于 M_dim 时，才算正交
        is_orthogonal = np.isclose(off_diagonal_sum, 0) and np.all(np.isclose(diagonal_values, M_dim))

        return is_orthogonal, off_diagonal_sum

# --- Tkinter GUI 应用程序类 ---

class HadamardGeneratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Hadamard 矩阵生成与子矩阵抽取工具")
        master.geometry("600x480")

        self.output_dir = os.getcwd()

        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill='both', expand=True)

        self.M_var = tk.StringVar(value="8") # M 阶数 (列数)
        self.N_var = tk.StringVar(value="4") # N 行数

        self._create_widgets(main_frame)

    def _create_widgets(self, frame):
        # 1. 输入区域
        input_frame = ttk.LabelFrame(frame, text="输入参数", padding="10")
        input_frame.pack(fill='x', pady=5)

        ttk.Label(input_frame, text="Hadamard 阶数 M (列数):").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.M_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(input_frame, text="抽取行数 N (生成 N x M 矩阵):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(input_frame, textvariable=self.N_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky='w')

        # 2. 输出路径
        ttk.Label(input_frame, text="输出目录:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.dir_label = ttk.Label(input_frame, text=self.output_dir, foreground='blue', wraplength=300)
        self.dir_label.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky='w')
        ttk.Button(input_frame, text="选择目录", command=self.select_directory).grid(row=2, column=3, padx=5, pady=5)

        # 3. 按钮
        btn_frame = ttk.Frame(frame, padding="5")
        btn_frame.pack(fill='x', pady=10)

        # 核心按钮：生成相位表
        ttk.Button(btn_frame, text="生成相位表 (生成 ZJ.csv & ZJMXN.csv)", command=self.generate_data_handler).pack(pady=10, fill='x')

        # 4. 日志输出区域
        log_frame = ttk.LabelFrame(frame, text="处理日志和结果", padding="10")
        log_frame.pack(fill='both', expand=True, pady=5)

        self.log_display = tk.Text(log_frame, height=12, wrap='word')
        self.log_display.pack(fill='both', expand=True)

        self.log_display.insert(tk.END, "程序已启动。请输入 M (2的幂次) 和 N (行数)，然后点击按钮。\n")

    def select_directory(self):
        """打开文件对话框选择保存目录"""
        new_dir = filedialog.askdirectory(initialdir=self.output_dir)
        if new_dir:
            self.output_dir = new_dir
            self.dir_label.config(text=self.output_dir)
            self.log_message(f"✅ 输出目录已设置为: {self.output_dir}")

    def log_message(self, message):
        """向日志区域追加消息"""
        self.log_display.insert(tk.END, message + "\n")
        self.log_display.see(tk.END) # 自动滚动到最新消息

    def generate_data_handler(self):
        """处理点击事件，执行生成和保存逻辑"""
        M_str = self.M_var.get()
        N_str = self.N_var.get()

        try:
            M = int(M_str)
            N = int(N_str)
        except ValueError:
            messagebox.showerror("输入错误", "M 和 N 必须是整数。")
            self.log_message("[❌ 错误] M 或 N 输入无效。")
            return

        # 参数校验
        if N > M:
            messagebox.showerror("输入错误", "N (抽取行数) 不能大于 M (Hadamard 矩阵阶数)。")
            self.log_message(f"[❌ 错误] N ({N}) 不能 > M ({M})。")
            return
        if M & (M - 1) != 0 or M < 1:
            messagebox.showerror("输入错误", "M 必须是 2 的幂次方 (1, 2, 4, 8, ...)。")
            self.log_message(f"[❌ 错误] M ({M}) 不是 2 的幂次方。")
            return

        self.log_display.delete('1.0', tk.END) # 清空旧日志
        self.log_message(f"--- 开始处理 M={M}, N={N} (抽取 N x M 矩阵) ---")

        try:
            # 1. 生成 M x M Hadamard 矩阵
            had_tool = HadamardMatrix(M)
            self.log_message(f"1. 成功生成 {M}x{M} Hadamard 矩阵。")

            # 2. 【最终修正】抽取 N x M 矩阵 (前 N 行，所有 M 列)
            # R=N (行数), C=M (列数)
            H_N_M = had_tool.extract_submatrix(R=N, C=M)
            self.log_message(f"2. 成功抽取 N x M ({N}x{M}) 子矩阵。")

            # --- 文件名和路径 ---
            base_dir = self.output_dir
            ZJ_FILENAME = os.path.join(base_dir, "ZJ.csv")
            ZJMXN_FILENAME = os.path.join(base_dir, "ZJMXN.csv")

            # 3. 保存 N x M 矩阵到 ZJ.csv (±1 矩阵)
            HadamardMatrix.save_to_csv(H_N_M, ZJ_FILENAME)
            self.log_message(f"3. ✅ 矩阵 (+1/-1) 已保存到: {ZJ_FILENAME}")

            # 4. 映射为角度并保存到 ZJMXN.csv (相位表/deg文件)
            H_N_M_DEG = HadamardMatrix.map_to_degrees(H_N_M)
            HadamardMatrix.save_to_csv(H_N_M_DEG, ZJMXN_FILENAME)
            self.log_message(f"4. ✅ **相位表 (deg)** 已保存到: {ZJMXN_FILENAME}")

            # 5. 验证正交性 (现在验证 N 个 M 维行向量的正交性)
            is_orthogonal, error_sum = HadamardMatrix.verify_orthogonality(H_N_M)

            ortho_msg = f"5. 正交性验证 (行向量维度={M}, 期望内积={M}):\n"

            if is_orthogonal:
                 ortho_msg += f"   - 结论: **完全正交**。非对角线元素总和: {error_sum:.4f}"
            else:
                 ortho_msg += f"   - 结论: **不正交**。非对角线元素总和: {error_sum:.4f}"

            self.log_message(ortho_msg)

            self.log_message("\n--- 处理成功！请查看输出目录 ---")

        except Exception as e:
            messagebox.showerror("处理错误", f"在生成或保存过程中发生错误: {e}")
            self.log_message(f"[❌ 致命错误] {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = HadamardGeneratorApp(root)
    root.mainloop()
