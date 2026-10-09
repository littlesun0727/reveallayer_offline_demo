# RevealLayer 离线元素提取实验

在已有 [RevealLayer](https://github.com/360CVGroup/RevealLayer) 仓库和模型环境中，根据 `analysis.json` 的 overlay 框提取透明图层。

## 运行

激活远程已有环境，在本仓库目录执行：

```bash
conda activate reveallayer

python run_overlays.py --check-only \
  --output /data/w00911743/dataset/tasks/16-海边人像拼图/reveallayer_check

CUDA_VISIBLE_DEVICES=0 python -u run_overlays.py \
  --ids starfish_pair doodle_spinner speech_bubble \
  --bbox-margin 8 --offload \
  --output /data/w00911743/dataset/tasks/16-海边人像拼图/reveallayer_smoke_s43
```

脚本默认配置：

- RevealLayer 源码：`/data/w00911743/RevealLayer`
- 模型搜索目录：`/data/w00911743/pretrained_models/reveallayer`
- 输入任务：`/data/w00911743/dataset/tasks/16-海边人像拼图`
- 输入文件：`prepared/reference.png` 和 `analysis.json`

模型目录需要完整 FLUX.1-dev 和 RevealLayer 权重。本仓库不包含模型或上游源码。
FLUX 在其他目录时使用 `--flux /实际路径/FLUX.1-dev`。
每次实验使用新的输出目录。省略 `--ids` 可处理全部 overlay。

主要结果为 `contact_sheet.jpg`、`elements/<id>/rgba.png`、`alpha.png` 和 `full_rgba.png`。

详见 [完整说明](README.txt)。本脚本通过本地坐标及文件检查测试，尚未完成远程 A100 推理验证。

## 本地预览

`local_prepare_crop/`、`local_prepare_full/`、`local_prepare_smoke/` 是参考图和目标框的输入预览，**不是模型抠图结果**。这些目录不参与远程推理，也不需要修改其中记录的本地路径。

## CPU 测试

安装 Pillow 和 NumPy 后：

```bash
python -m unittest -v test_geometry.py
```
