# Low-Diffusion Dipole Explorer

这是一个从 CygBubble 项目中抽出的独立交互式程序，用于研究均匀有序磁场中的球形低扩散区如何改变背景宇宙线偶极的振幅、方向和赤经相位。
![Explorer preview](assets/explorer_preview.png)

## 功能

- 交互调整背景相对梯度的银经、银纬和模长；
- 交互调整有序磁场方向、阿尔芬马赫数和低扩散系数比例；
- 使用固定银河坐标中的 `r/R, l, b` 指定观测点；
- 绘制银道天图、解析解二维切片、偶极振幅和赤经相位；
- 支持操作历史和单参数连续扫描；
- 在扫描 `D_low/D_parallel` 时显示等效能量双横轴；
- 可选叠加 Li et al. (2024) Figure 3 的实验振幅与相位数据。

## 环境

- Python 3.10 或更高版本
- NumPy
- Matplotlib

安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 中可用：

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 运行

在仓库根目录执行：

```bash
python LowDiffusionDipoleExplorer.py
```

也可以在 PyCharm 中直接运行 `LowDiffusionDipoleExplorer.py`。程序需要图形桌面；无桌面的 Linux 服务器需要 X11 转发、VNC 或其他图形会话。

## 观测数据叠加

1. 将 `Curve mode` 设为 `parameter scan`；
2. 将扫描参数设为 `Dlow / Dparallel`；
3. 勾选窗口顶部的 `Li+2024 data (E scan)`。

实验数据的能量通过

```text
a(E) = A_at_10TeV * (E / 10 TeV) ** Delta_low_parallel
```

映射到主横轴。只有这个能量换算关系有效，并且正在扫描 `Dlow / Dparallel` 时，数据点才会显示。观测振幅是赤经一阶谐波振幅，应主要与图中的 `RA-projected amplitude` 比较。

## 参数位置

- 窗口分辨率、解析切片精度、扫描点数和振幅纵轴范围：`LowDiffusionDipoleExplorer.py` 顶部；
- 扩散系数、能量幂律指数和坐标投影矩阵：`cygbubble/config.py`；
- Li et al. 数据：`data/Li_2024_Figure3_vector_digitized.csv`。

## 文件结构

```text
.
├── LowDiffusionDipoleExplorer.py       # 动态窗口入口
├── cygbubble/
│   ├── config.py                       # 独立程序所需物理常量
│   ├── coordinates.py                  # 银道/赤道/笛卡尔坐标转换
│   ├── observations.py                 # Li et al. CSV 读取与校验
│   ├── analysis/
│   │   └── low_diffusion_dipole.py     # 偶极方向和真实振幅计算
│   ├── analytic/
│   │   └── sphere.py                   # 球形低扩散区解析解
│   └── physics/
│       └── diffusion.py                # 能量与扩散系数换算
├── data/
│   └── Li_2024_Figure3_vector_digitized.csv
├── tests/
│   └── test_smoke.py
├── assets/
│   └── explorer_preview.png
└── requirements.txt
```

## 验证

```bash
python -m unittest discover -s tests
```

