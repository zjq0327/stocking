# 周期平针样片的准静态拉伸

这一版把 Crane 参数曲线作为初始纱线路径，通过简化的杆模型求解周期样片的平衡形状，再重建管面并重新烘焙双面的 ID/P/N/T。它是独立实现的研究演示，不是 HYLC 原代码移植、论文复现或经过实测标定的丝袜材料模型。

## 几何、材料和加载分别控制什么

现有几何参数 `a`、`h`、`d`、`rowOffset`、`R` 与 `scale_mm` 继续描述初始线圈形状和实际尺寸。它们不等于弹性模量。初始周期为

\[
L_x^0=2\pi\,\mathrm{scale\_mm},\qquad
L_y^0=\mathrm{rowOffset}\,\mathrm{scale\_mm}.
\]

物理管半径固定为 \(r=R\,\mathrm{scale\_mm}\)。本版不模拟截面压扁、纱线伸长造成的变细、包覆纱内部结构或单根纤维。

拉伸配置使用独立的 `StretchParameters`：`axial_stiffness_N` 表示轴向刚度 \(EA\)，单位 N；`bending_stiffness_N_mm2` 表示弯曲刚度 \(B\)，单位 N·mm²；`contact_stiffness_N_per_mm` 控制接触屏障的强度，单位 N/mm。长度统一为 mm，能量为 N·mm。这些数值是演示值，不能解读为某种尼龙/氨纶丝袜的测量数据。这里允许独立选择 \(EA\) 和 \(B\)，不强制它们来自同一根均匀实心圆柱的杨氏模量。

`lambda_x`、`lambda_y` 是指定的周期长度比：

\[
L_x=\lambda_xL_x^0,\qquad L_y=\lambda_yL_y^0.
\]

两者都限制在 [1, 1.6]。`stretch-x.json` 指定 (1.2, 1.0)，即横向增加 20% 且纵向周期保持原值；这不是侧边自由收缩的单轴拉伸试验。`stretch-y.json` 指定 (1.0, 1.2)，`stretch-biaxial.json` 指定 (1.15, 1.15)。自由横向收缩和压缩加载尚未实现。

## 材料参考态与加载路径

1. 从初始周期曲线离散出杆节点，保存每段材料原长 \(\ell_i^0\)。采用自然直杆，即自然曲率为零。
2. 在原周期 \((L_x^0,L_y^0)\) 下求解一次平衡。原长保持不变，周期边界和编织拓扑会限制纱线完全变直，因此这个样片参考状态可能仍含弯曲能、接触力和边界反力。
3. 通过 `load_steps` 逐步改变周期长度，并以前一步形状作为下一步的初始猜测。始终保留步骤 1 的材料原长和自然曲率；不能每一步都把当前形状重新设成材料无应力状态。
4. 以最终中心线重建周期邻居、管面与标架，重新计算 ID/P/N/T。形变会改变遮挡和孔隙，不能仅缩放旧贴图。

“原周期下的平衡样片”“材料原长/自然曲率”和“几何初始猜测”是三个不同概念。观察到样片已静止，也不能据此推断其内部应力为零。

本版固定用户指定的参考针距、行距，不进一步优化自由周期尺寸。因此某些加载过程可能释放已有弯曲能，使总能量下降；不能把这条加载路径直接解释为从自由、无应力织物出发测得的拉伸刚度。若要拟合布面本构或真实穿着张力，需要另行确定自然周期和预应力，并用实测数据标定。

## 简化杆能量和求解

令 \(\mathbf e_i=\mathbf x_{i+1}-\mathbf x_i\)、\(\ell_i=\|\mathbf e_i\|\)、\(\mathbf t_i=\mathbf e_i/\ell_i\)。跨越周期边界的边需要加入相应周期平移；它不能被误读为跳回原点的一条短边。

轴向能为

\[
E_s=\sum_i \frac{EA}{2\ell_i^0}(\ell_i-\ell_i^0)^2.
\]

有限的轴向刚度只会惩罚伸长，不会产生数学上严格的不可伸长约束。`max_strain` 是允许的纱线应变诊断阈值，与样片的周期伸长比不同。

自然直杆的离散弯曲采用曲率二项量

\[
\mathbf k_i=\frac{2\,\mathbf t_{i-1}\times\mathbf t_i}
{1+\mathbf t_{i-1}\cdot\mathbf t_i},\qquad
\bar\ell_i^0=\frac{\ell_{i-1}^0+\ell_i^0}{2},\qquad
E_b=\sum_i\frac{B}{2\bar\ell_i^0}\|\mathbf k_i\|^2.
\]

零长度边和相邻切线反向属于退化情形，不能作为正常收敛结果接受。该模型只优化中心线，不包含独立的材料扭转角、扭转能或摩擦历史。

总能量是 \(E_s+E_b+E_c\)。接触项使用非相邻杆线段的最近距离 \(d_{ij}\)，包括两端点和两条线段内部的候选最近点；周期邻居中的无序线段对只计算一次。同一根纱线上按材料原长计算的局部弧长间隔不超过 \(\pi r\) 的线段对排除，以免把相连圆管自身的局部连接当成碰撞。

令 \(g_{ij}=d_{ij}-2r\)、\(h=\mathrm{contact\_margin\_ratio}\,r\)、\(w_{ij}=\ell_i^0\ell_j^0/(2r)^2\)，则

\[
E_c=\sum_{ij}w_{ij}\,b(g_{ij}),\qquad
b(g)=\begin{cases}
-k_c(g-h)^2\log(g/h),&0<g<h,\\
0,&g\ge h.
\end{cases}
\]

\(k_c\) 对应 `contact_stiffness_N_per_mm`。当 \(g\le0\) 时该状态直接被拒绝；屏障不是允许纱线截面压缩的材料规律。权重使接触刚度不至于仅因增加离散节点数就成倍增大，但仍需要分辨率收敛检查。

**步长还要通过连续运动检查。** `path_clear` 对整段线性节点运动使用距离的 Lipschitz 下界，并对子时间区间递归细分；它同时检查加载预测步和优化试探步。不能在最大细分深度内判定安全的区间会被拒绝。因此此处不只检查两个端点，也不是用有限时间采样代替运动过程。初始不可行状态、零间隙及不安全加载预测会终止求解。

这项检查的范围是本实现枚举到的非局部周期胶囊线段对，使用浮点运算；局部弯折和后续管面重建还需另行检查。它不是 C-IPC/ACCD 的复现，也不应被扩展表述为对连续光滑纱线、任意截面或最终三角网格的普适无穿透证明。

使用自行实现的 **L-BFGS 双循环递推**逐级求静态平衡，最多保留 30 对位置/梯度差分，不含惯性演化。采用可行 Armijo 回溯线搜索：试探点必须满足能量充分下降和整步接触检查；不合法或不能判定安全的试探会缩短步长重试，不能作为零梯度的有效状态接受。回溯无法前进且残余力仍未达标时中止。这与 HYLC 官方重实现中的受约束 Newton 求解器、其周期自由度处理及参数滑移约束不同。节点平均位置保持固定以消除整体平移自由度，并未复制 HYLC 的每根周期纱线滑移约束。

每一阶段都独立检查最大节点残余力不超过 `gradient_tolerance`（单位 N），以及最大绝对轴向应变不超过 `max_strain`。不满足条件就中止；不能仅凭优化器状态显示成功继续生成资产。`max_iterations` 是每阶段迭代上限。这些属于求解与验收参数，不是材料参数。

`nodes` 是力学离散分辨率，范围 32–256；它与管面周向细分、最终烘焙图尺寸分开。提高离散分辨率用于检查结果是否稳定，不能弥补缺失的材料规律。

## 运行依赖

静态几何、烘焙和拉伸求解均使用 Blender 自带的 Python 与 NumPy。L-BFGS 更新和线搜索直接实现在 `stretch_solver.py`，无需另装 SciPy 或其他优化器。本版也不需要编译 HYLC，不依赖它的 Eigen、pybind11、TBB 或图形界面组件。

## 标架与双面数据

变形后的中心线不再保持 Crane 解析曲线的形式。重建管面时使用沿切线的平行运输标架，并核对周期接缝连续性。Bishop 标架表示一种几何框架，不等于已经模拟了真实纱线扭转。中心线有向切线用于纱线方向 T，外向管面法线用于 N；P 来自重新生成的表面。正反面的可见交点需要分别计算，不能简单对背面 N/T 全部取负。

## 文献依据与本版边界

- [Nocedal，L-BFGS 官方资料与参考文献](https://users.iems.northwestern.edu/~nocedal/lbfgs.html)：Nocedal 1980、Liu 与 Nocedal 1989 的有限记忆拟牛顿方法是本版优化器的算法背景；可行 Armijo 回溯及杆接触约束是本项目的实现选择，没有复制该页面发布的 Fortran 软件。
- [Bergou 等，Discrete Elastic Rods，2008](https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf)：§3/Algorithm 1 区分初始状态和材料参考态；§4.1.1 与 §4.2.1 给出弯曲能和离散曲率；§4.1.2、§4.2.2 给出 Bishop 标架和平行运输。原文同时包含材料扭转，本版未完整实现 DER。原文的“quasistatic”主要用于材料标架，不能据此声称其中心线本来就使用本版的静态 L-BFGS 流程。
- [Kaldor 等，Simulating Knitted Cloth at the Yarn Level，2008](https://www.cs.cornell.edu/~srm/publications/SG08-knit-lr.pdf)：§3.2 描述线圈重排及纱线伸长的作用；§4.1 给出纱长约束；§4.2 使用接触罚能。该论文不提供本版数值参数对弹性丝袜的标定。
- [Leaf 等，Interactive Design of Periodic Yarn-Level Cloth Patterns，2018](https://graphics.stanford.edu/projects/yarnsim/assets/interactive_design_of_periodic_yarn_level_cloth_patterns.pdf)：§5 是周期边界；§7（论文页 202:9）通过改变横纵重复距离拉伸/压缩；§3.1 使用有限伸长能和阻尼动力学松弛。该流程支持周期样片思路，但不等于本版求解器的逐行算法。
- [Sperl 等，Homogenized Yarn-Level Cloth，2020](https://visualcomputing.ist.ac.at/publications/2020/HYLC/)：补充 §S2.1–S2.3 和伪代码 C1–C4 分别说明周期处理、静态优化、参考态和变形采样。[官方源码](https://git.ista.ac.at/gsperl/HYLC) 提供更完整的周期杆优化和 Python 接口。本版未移植其 C++ 求解器，也没有采用 C1 的经验预张紧系数与自由周期优化。
- [Sperl 等，Mechanics-Aware Deformation of Yarn Pattern Geometry，2021](https://pub.ista.ac.at/group_wojtan/projects/2021_Sperl_MADYPG/2021_MADYPG_paper_lowres.pdf)：§3.1、Fig.3 根据宏观变形边界条件优化周期纱线的弹性静态平衡，为制作时调整纱线形状提供进一步参考。
- [Zhu、Yan 等，A Realistic Surface-based Cloth Rendering Model，2023](https://sites.cs.ucsb.edu/~lingqi/publications/paper_sig23cloth.pdf)：§4.1 定义双面 ID/P/N/T 几何特征。本项目把变形几何重新投影到该输入表示，并未实现整篇论文的光学模型。
- [Poincloux 等，Geometry and Elasticity of a Knitted Fabric，作者预印本](https://arxiv.org/pdf/1801.08355)：第 3 页讨论参考状态中的内应力；§II.B 与 §IV 解释细纱、松针织下忽略纱线伸长的条件，以及锁紧后该近似的失效。这里引用的是预印本章节编号。
- [Li 等，Codimensional Incremental Potential Contact，2021](https://ipc-sim.github.io/C-IPC/file/paper_small.pdf)：§5、§5.4 和 Algorithm 1 将厚度约束与连续碰撞检测结合。引用它用于说明严格接触需要哪些条件，不表示本版实现了 C-IPC。

本版适合观察理想平针单元在指定周期伸长下的几何重排。真实弹性丝袜还需要纱线拉伸曲线、预拉伸/热定形、包覆结构、接触压缩与摩擦等信息；未加入这些信息之前，不应把输出当成对透肤变化、压力或回弹的定量预测。
