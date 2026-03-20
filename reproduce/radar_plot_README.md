# Radar Plot Script

这个脚本用于复刻你给出的那种多模型论文风格雷达图。

## 1. 输入格式

支持两种输入：

### JSON

推荐用 JSON，因为可以控制颜色、线型、填充和图例位置。
现在也支持控制：

- 百分比刻度
- 图例框透明度和边框
- 网格线颜色
- 外圈边框粗细
- 轴标签和刻度字号

示例文件：

`D:\pywork\MiniRAG1333213213\reproduce\radar_plot_example.json`

### CSV

适合快速画图。表头格式如下：

```csv
series,1-hop,2-hop,3-hop,5-hop,10-hop,Main connected component
HippoRAG,0.06,0.15,0.21,0.29,0.34,0.36
AriGraph,0.05,0.12,0.15,0.20,0.21,0.21
Wikontic,0.07,0.22,0.31,0.42,0.48,0.50
```

CSV 只负责数值，不负责单条曲线样式。

## 2. 运行方式

```bash
python reproduce/radar_plot.py reproduce/radar_plot_example.json -o reproduce/result/radar_example.png
```

如果你的环境里命令不是 `python`，改成你自己的 Python 可执行文件即可。

你的真实数据也已经整理好一份：

`D:\pywork\MiniRAG1333213213\reproduce\reachability_radar_real.json`

直接画图：

```bash
python reproduce/radar_plot.py reproduce/reachability_radar_real.json -o reproduce/result/reachability_radar_real.png
```

这份 `JSON` 已经预设成更接近论文图的样式：

- 白底
- 浅灰网格
- 细外边框
- 右下角图内图例
- 百分比径向刻度
- 更克制的填充透明度

这份真实数据目前已经额外包含一列 `AEC`：

- `LightRAG = 0.939`
- `MiniRAG = 0.939`
- `HybridRAG = 0.962`

如果以后你又产出新的 `*_all_reachability_report.json`，可以直接自动转换：

```bash
python reproduce/reachability_reports_to_radar.py ^
  "D:\学习\学\研究生\毕设\实验结果\mini_k1_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\low_k1_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\hybrid_k1_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\mini_k2_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\low_k2_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\hybrid_k2_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\mini_k3_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\low_k3_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\hybrid_k3_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\mini_k5_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\low_k5_all_reachability_report.json" ^
  "D:\学习\学\研究生\毕设\实验结果\hybrid_k5_all_reachability_report.json" ^
  --extra-metric AEC:mini:0.939 ^
  --extra-metric AEC:low:0.939 ^
  --extra-metric AEC:hybrid:0.962 ^
  -o reproduce/result/reachability_radar.json
```

## 3. JSON 字段说明

- `categories`: 雷达图每个轴的名称
- `style.r_max`: 半径最大值
- `style.r_ticks`: 同心圆刻度
- `series[].name`: 图例名称
- `series[].values`: 与 `categories` 一一对应的数值
- `series[].color`: 线条颜色
- `series[].linestyle`: 线型，支持 `-`、`--`、`-.`、`:`
- `series[].fill`: 是否填充
- `series[].fill_alpha`: 填充透明度
- `series[].linewidth`: 单条线宽

## 4. 依赖

脚本依赖：

```bash
pip install matplotlib numpy
```
