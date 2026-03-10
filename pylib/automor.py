import numpy as np
from enum import Enum, auto

# ==========================================
# 1. 基础数据结构：枚举与代数群
# ==========================================

class Pauli(Enum):
    """泡利算子枚举"""
    I = auto()
    X = auto()
    Y = auto()
    Z = auto()

class AutomorphismData:
    """平移自同构（Shift Automorphism）数据结构，Z6 x Z6 群"""
    def __init__(self, x: int, y: int):
        self.x = x % 6
        self.y = y % 6

    def inv(self) -> 'AutomorphismData':
        """求逆：模 6 空间中的加法逆元"""
        return AutomorphismData(6 - self.x, 6 - self.y)

class TwoBases:
    """逻辑比特的初始基底 (代表前 6 个和后 6 个逻辑比特)"""
    def __init__(self, basis_1: Pauli, basis_7: Pauli):
        self.basis_1 = basis_1
        self.basis_7 = basis_7

class NativeMeasurement:
    """原生测量任务：包含目标基底和硬件执行的平移路线"""
    def __init__(self, logical: TwoBases, automorphism: AutomorphismData):
        self.logical = logical
        self.automorphism = automorphism

# ==========================================
# 2. 核心编译引擎：机器码生成器
# ==========================================

class CodeMeasurement:
    def __init__(self, mx: np.ndarray, my: np.ndarray):
        self.mx = mx
        self.my = my

    def _mat_pow_mod2(self, mat: np.ndarray, power: int) -> np.ndarray:
        """辅助函数：在 F_2 (模 2) 空间下计算矩阵的幂次"""
        res = np.eye(mat.shape[0], dtype=int)
        base = mat.copy()
        while power > 0:
            if power % 2 == 1:
                res = (res @ base) % 2
            base = (base @ base) % 2
            power //= 2
        return res

    def measures(self, native_measurement: NativeMeasurement) -> np.ndarray:
        """
        核心翻译引擎：将物理平移 + 逻辑意图，翻译为 24 位的机器码 (PauliString)
        """
        # 在 nalgebra 中，Vector6::identity() 生成的是对角线为 1 的 6x1 向量
        # 即 [1, 0, 0, 0, 0, 0]^T，物理意义：绝对精准地瞄准第一个逻辑量子比特
        one = np.array([1, 0, 0, 0, 0, 0], dtype=int)
        zero = np.array([0, 0, 0, 0, 0, 0], dtype=int)

        # 闭包 1：获取逻辑意图对应的辛坐标 (x, z)
        def get_pauli_vectors(basis: Pauli):
            if basis == Pauli.I: return zero, zero
            if basis == Pauli.X: return one, zero
            if basis == Pauli.Z: return zero, one
            if basis == Pauli.Y: return one, one
            raise ValueError("Unknown Pauli basis")

        x1, z1 = get_pauli_vectors(native_measurement.logical.basis_1)
        x7, z7 = get_pauli_vectors(native_measurement.logical.basis_7)

        # 闭包 2：计算硬件物理平移带来的总变换矩阵 (等价于 Rust 的 action 闭包)
        def action(a: AutomorphismData) -> np.ndarray:
            mx_pow = self._mat_pow_mod2(self.mx, a.x)
            my_pow = self._mat_pow_mod2(self.my, a.y)
            return (mx_pow @ my_pow) % 2

        # 辛几何不对称映射：X 乘逆矩阵，Z 乘转置矩阵！(核心灵魂)
        x_action = action(native_measurement.automorphism.inv())
        z_action = action(native_measurement.automorphism).T

        # 开火：矩阵乘以初始坐标，并严格限制在模 2 空间
        map_x1 = (x_action @ x1) % 2
        map_x7 = (x_action @ x7) % 2
        map_z1 = (z_action @ z1) % 2
        map_z7 = (z_action @ z7) % 2

        # 叠罗汉：将 4 个长度为 6 的向量拼接成 24 位的机器码掩码 (PauliString)
        result = np.concatenate([map_x1, map_x7, map_z1, map_z7])
        
        return result

# ==========================================
# 3. 硬件白皮书参数硬编码
# ==========================================

GROSS_MEASUREMENT = CodeMeasurement(
    mx = np.array([
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1],
    ], dtype=int),
    my = np.array([
        [1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0, 1],
    ], dtype=int)
)


def _simple_self_check() -> None:
    """Run a minimal deterministic check and print the result."""
    # Case 1: identity shift should keep X@1 and Z@7 anchors unchanged.
    task = NativeMeasurement(
        logical=TwoBases(Pauli.X, Pauli.Z),
        automorphism=AutomorphismData(0, 0),
    )
    result = GROSS_MEASUREMENT.measures(task)

    expected = np.zeros(24, dtype=int)
    expected[0] = 1    # X on logical qubit 1 anchor
    expected[18] = 1   # Z on logical qubit 7 anchor

    assert result.shape == (24,), f"Unexpected shape: {result.shape}"
    assert np.all((result == 0) | (result == 1)), "Output is not binary over F2."
    assert np.array_equal(result, expected), f"Identity check failed:\n{result}\n!=\n{expected}"

    # Case 2: another shift should still produce a valid 24-bit binary vector.
    task2 = NativeMeasurement(
        logical=TwoBases(Pauli.Y, Pauli.I),
        automorphism=AutomorphismData(1, 2),
    )
    result2 = GROSS_MEASUREMENT.measures(task2)
    assert result2.shape == (24,), f"Unexpected shape in case 2: {result2.shape}"
    assert np.all((result2 == 0) | (result2 == 1)), "Case 2 output is not binary over F2."

    print("Self-check passed.")
    print("case1 (identity):", result.tolist())
    print("case2 (shift 1,2):", result2.tolist())


# ==========================================
# 4. 逻辑算子演化观测器
# ==========================================

def decode_logical_pauli(arr24: np.ndarray) -> str:
    """
    将 24 位的数组解码为人类可读的逻辑泡利字符串。
    例如：输出 "X_4 ⊗ X_6"
    """
    terms = []
    for i in range(12):
        # 提取第 i+1 个逻辑比特的 X 位和 Z 位
        x_bit = arr24[i]
        z_bit = arr24[i + 12]
        
        idx = i + 1  # 逻辑比特通常从 1 开始编号
        
        if x_bit == 1 and z_bit == 1:
            terms.append(f"Y_{idx}")
        elif x_bit == 1:
            terms.append(f"X_{idx}")
        elif z_bit == 1:
            terms.append(f"Z_{idx}")
            
    if not terms:
        return "I (恒等)"
    return " ⊗ ".join(terms)

def observe_shift_evolution(dx: int, dy: int):
    """
    观测特定的位移 (dx, dy) 对基础逻辑算子产生的形变影响
    """
    print(f"\n--- 观测位移 Automorphism(dx={dx}, dy={dy}) 的代数影响 ---")
    
    # 我们测试 4 种基础的起始状态
    test_cases = [
        ("X_1", TwoBases(Pauli.X, Pauli.I)),
        ("Z_1", TwoBases(Pauli.Z, Pauli.I)),
        ("X_7", TwoBases(Pauli.I, Pauli.X)),
        ("Z_7", TwoBases(Pauli.I, Pauli.Z)),
    ]
    
    for name, basis in test_cases:
        # 1. 获取未位移时的原始状态 (Identity)
        task_id = NativeMeasurement(logical=basis, automorphism=AutomorphismData(0, 0))
        res_id = GROSS_MEASUREMENT.measures(task_id)
        str_id = decode_logical_pauli(res_id)
        
        # 2. 获取位移后的演化状态
        task_shift = NativeMeasurement(logical=basis, automorphism=AutomorphismData(dx, dy))
        res_shift = GROSS_MEASUREMENT.measures(task_shift)
        str_shift = decode_logical_pauli(res_shift)
        
        print(f"初始逻辑态: {str_id:<5}  ==经过位移==>  演化为: {str_shift}")

# 替换原本的 main 执行块
if __name__ == "__main__":
    _simple_self_check()
    
    # 案例 A：向东平移 1 格 (论文中提到的基础生成元)
    observe_shift_evolution(dx=1, dy=0)
    
    # 案例 B：向东向北各平移 6 格 (理论上应该是 Identity 恒等映射)
    observe_shift_evolution(dx=6, dy=6)