# SymmMatmul Example Readme

## 功能说明

- 算子功能：完成对称矩阵乘法计算（SYMM），利用输入矩阵的对称性，仅读取上三角（或下三角）有效数据，通过 direct/transpose 双路径完成等价的全量矩阵乘。
- 计算公式：

  左侧对称矩阵乘（`symmSide=0`，要求 `M == K`）：

  $$
  C = S \times B
  $$

  其中 $S$ 是 $M \times M$ 对称方阵（$S = S^T$），$B$ 是 $K \times N$ 的普通矩阵。

  右侧对称矩阵乘（`symmSide=1`，要求 `K == N`）：

  $$
  C = B \times S
  $$

  其中 $B$ 是 $M \times K$ 的普通矩阵，$S$ 是 $N \times N$ 对称方阵。

- 支持上三角（`symmFill=1`）和下三角（`symmFill=0`）两种有效数据区域。

## 参数说明

以下是本样例的命令行参数：

| 参数名 | 描述 | 默认值 |
| ----- | -------- | ------ |
| `m` | 输出矩阵 C 的行数 | 必填 |
| `n` | 输出矩阵 C 的列数 | 必填 |
| `k` | GEMM 归约维度 | 必填 |
| `deviceId` | 使用的 NPU 卡 ID | `0` |
| `symmSide` | 对称矩阵所在侧：`0`=LEFT（C=S×B），`1`=RIGHT（C=B×S） | `1` |
| `symmFill` | 有效三角区域：`1`=UPPER（上三角），`0`=LOWER（下三角） | `1` |

### Shape 约束

| 模式 | 约束 | 原因 |
|------|------|------|
| LEFT（`symmSide=0`） | `M == K` | 左侧操作数必须是方阵 |
| RIGHT（`symmSide=1`） | `K == N` | 右侧操作数必须是方阵 |

### 支持的数据类型

当前示例固定使用 `float`（fp32）：

| ElementA | ElementB | ElementC | Layout |
|----------|----------|----------|--------|
| `float` | `float` | `float` | `RowMajor` |


## 代码组织

```
├── 75_symm
│   ├── CMakeLists.txt     # CMake 编译文件
│   ├── README.md
│   └── symm.cpp    # 主文件（包含 host 数据生成、kernel 调度、正确性验证）
```


## 使用示例

1. 编译样例代码：

```bash
bash scripts/build.sh 75_symm
```

2. 执行算子样例程序：

```bash
cd output/bin

# 左乘 + 上三角：C = S(M×M) × B(M×N)，S 上三角有效
./75_symm 768 4096 768 0 0 1

# 左乘 + 下三角
./75_symm 768 4096 768 0 0 0

# 右乘 + 上三角：C = B(M×K) × S(K×K)，S 上三角有效
./75_symm 4096 768 768 0 1 1

# 右乘 + 下三角
./75_symm 4096 768 768 0 1 0
```

参数含义依次为：`m n k deviceId symmSide symmFill`。

3. 执行成功输出：

```
Compare success.
```
