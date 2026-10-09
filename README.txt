RevealLayer 离线 overlay 提取实验
================================

脚本：run_overlays.py
用途：在已有 RevealLayer 仓库和 conda 环境中，对 analysis.json 的 objects[] 中
kind == "overlay" 的元素逐个提取；不会更改上游仓库、原图或 analysis.json。
依据：2026-10-09 查看 360CVGroup/RevealLayer main 分支源码。
未连接远程 A100，尚未验证 GPU 推理和真实 checkpoint 兼容性。

一、复制文件
把本目录中的 run_overlays.py 复制到远程：
  /data/w00911743/RevealLayer/run_overlays.py
无需把本地生成的 local_prepare_* 目录传到远程。

二、模型要求
--models 默认为 /data/w00911743/pretrained_models/reveallayer。
脚本递归查找下面三个标志文件；存在多份时要求用 --flux / --ckpt / --decoder 明确选择。
一个典型布局（不是要求改成这个布局）：

  reveallayer/
    FLUX.1-dev/
      model_index.json
      transformer/       配置 + 全部分片权重
      vae/               配置 + 权重
      text_encoder/      配置 + 权重
      text_encoder_2/    配置 + 全部分片权重
      tokenizer/         配置 + 词表
      tokenizer_2/       配置 + 词表
      scheduler/         scheduler_config.json
    RevealLayer/
      pytorch_lora_weights.safetensors
      layer_pe.pt        或 layer_pe_corrected.pt，后者优先
      Refiner.pt
      xvae/transparent_decoder_ckpt.pth

如果只有 RevealLayer 的几个 checkpoint，没有完整 FLUX.1-dev，也没有它的两个
文本编码器和 tokenizer，不能离线运行；须先在可联网机器取得完整模型，再传过去。
FLUX 在另一个位置时，加 --flux /实际路径/FLUX.1-dev 即可。
检查会识别缺失分片、空文件、Git LFS 指针。实际模型内容是否兼容由严格加载验证。

三、先检查路径（不加载 GPU 模型）

  conda activate reveallayer
  cd /data/w00911743/RevealLayer
  python run_overlays.py --check-only \
    --output /data/w00911743/dataset/tasks/16-海边人像拼图/reveallayer_check

只想检查坐标、尚无权重：把 --check-only 改成 --prepare-only。
这两种模式仍会输出输入图和框预览。每次 --output 必须是新的或空目录。

四、先测三个代表元素

  CUDA_VISIBLE_DEVICES=0 python -u run_overlays.py \
    --ids starfish_pair doodle_spinner speech_bubble \
    --bbox-margin 8 --offload \
    --output /data/w00911743/dataset/tasks/16-海边人像拼图/reveallayer_smoke_s43

默认 resolution=1024，steps=30，guidance=1.0，seed=43。
--offload 用 Accelerate 把模型分阶段在 CPU/GPU 间移动，降低显存占用，但可能更慢。
A100 容量未确认，首测建议保留。显存充裕（例如 80GB）可去掉再比较。
这不是实测的显存保证；长宽、元素数和环境版本都会影响显存。
构造自定义 transformer 时 CPU 内存会同时暂存基础与自定义模型，主机内存也要充足。

五、测全部 13 个 overlay（模型只加载一次，按元素顺序推理）

  CUDA_VISIBLE_DEVICES=0 python -u run_overlays.py --bbox-margin 8 --offload \
    --output /data/w00911743/dataset/tasks/16-海边人像拼图/reveallayer_all_s43

六、整图分组对照

  CUDA_VISIBLE_DEVICES=0 python -u run_overlays.py --mode full --group-size 4 --offload \
    --output /data/w00911743/dataset/tasks/16-海边人像拼图/reveallayer_full_s43

整图模式始终读取原始 reference，每组最多 10 个目标；不是上一轮擦除后再提取。
默认每组 4 个，13 个目标共 4 次推理。各组的 background 只代表该组目标被移除，
不能作为“全部 overlay 已移除”的背景。crop 模式的 background 也只是各自局部背景。
不要把 max_layer_num 改成 15 来绕过上游限制：layer_pe 的 checkpoint 维度固定。

七、看结果

  contact_sheet.jpg
    左：参考图对应区域；中：生成 RGBA 放在棋盘背景；右：生成 RGBA 放在深色背景。
  boxes.png
    在原始参考图上画出的目标框，可检查 analysis 坐标是否遗漏元素边缘。
  passes/00/input.png, boxes.png, geometry.json
    真正输入模型的图、对齐后的框、裁剪缩放与 padding 映射。
  passes/00/background_rgba.png
    该次推理的背景层（模型分辨率）。
  passes/00/conditioning_reconstruction.png
    第 0 层对应输入条件，不是目标元素；目标从第 2 层开始。
  elements/<id>/rgba.png
    工作 bbox 尺寸的 RGBA，可直接观察元素边缘；会裁掉该 bbox 外的输出。
    指定 --bbox-margin 时工作 bbox 会扩张，metadata 同时保留 analysis_bbox 原标注。
  elements/<id>/alpha.png
    该 bbox 的灰度 alpha，软边缘保留，不做阈值二值化。
  elements/<id>/full_rgba.png
    映射回整张 reference 大小的透明图层，保留对齐扩框的输出。
  elements/<id>/model_rgba.png
    模型原始输出，未经坐标还原。
  elements/<id>/source_rgb_with_alpha.png
    对照实验：原参考图 RGB + 生成 alpha。不是 RevealLayer 原生 RGB；
    有助判断重绘程度，但不能还原被遮挡内容，也可能保留原背景污染和边缘杂色。
  DONE.json
    只有全部推理和导出成功才生成。程序异常会直接报错，不把失败当成功。

八、与官方入口的差异及理由

1. 只读取 overlay，不把照片窗和底图当目标。
2. 小元素逐个裁剪，默认每侧留 35% bbox 尺寸上下文，最小裁剪 128px。
   原图之外使用边缘延展。--context、--min-crop 可调整。
3. resize 后 pad 到 16 的倍数，完整保存坐标映射；不使用官方中心裁掉边缘的处理。
4. 优先使用仓库自带 diffusers/src，所有模型路径只读本地，禁用 HF 在线下载。
5. 不读取含作者绝对路径的 configs/base.py；直接使用 --flux 的基础配置。
6. 透明解码器 ViTEncoder 原本先读取作者路径下的 ImageNet ViT 权重，再加载解码器。
   脚本只在内存中移除这段 bootstrap，不改变结构；随后 strict=True 加载完整解码器。
   若缺少参数就停止，因此不会悄悄保留随机权重。源代码构造函数变化时也会提示复查。
7. LoRA、layer_pe、Refiner、透明解码器加载失败直接抛错。
8. 不使用官方 infer.sh；它默认启动 8 个 GPU 进程，且写死作者的 Python 路径。

九、效果预期

RevealLayer 是生成式分层模型，输出可以补全或改写 RGB，并非逐像素无损抠图。
建议先判断海星、气泡，再看细小放射花。白底雨滴卡片主要是完整矩形设计单元，
普通裁剪也可作为简单对照；条纹栏有被人物遮住的部分，补全是否正确需要目视判断。
white_drops 的一个大框中包含人物和照片，模型未必知道只要几个白色小水滴。
如果这组失败，可另外标每滴的小框分别测；不能由大框自动保证语义选择。
细英文文字也可能被改写、遗漏、糊掉，不应仅凭背景透明就判成功。
analysis 的框也不是精细 mask：已在预览发现大海星下尖超出原框，因此推荐命令加
--bbox-margin 8，对目标每侧扩 8 个原图像素；它与只增加上下文的 --context 不同。
参数默认 0，原 JSON 不变。工作 bbox 和原标注都记录在输出 metadata 中。

十、离线环境排错

ModuleNotFoundError / 版本错误：已有 conda 环境仍需包含上游依赖和自带 diffusers。
上游 requirements.txt 当前末尾有带 'pip install' 的行，不宜未经检查直接 pip -r。
不要在无网机器直接 pip install 在线源；缺包时在匹配 Linux/Python/CUDA 环境准备 wheelhouse，
再使用 pip install --no-index --find-links /wheelhouse ...。
源码要求的 flash-attn / xformers 编译包需匹配实际 torch/CUDA/Python。
CUDA OOM：先用 --offload，保持 crop 单目标；再测试 --resolution 768 或 512，
注意分辨率下降可能损失细字和小线条。不同设置使用不同输出目录。
透明解码器 strict load 失败：检查拿到的是否完整发布 checkpoint 和代码对应版本；
不要改成 strict=False 来掩盖缺参数。

来源：
https://github.com/360CVGroup/RevealLayer
https://github.com/360CVGroup/RevealLayer/blob/main/infer.py
https://github.com/360CVGroup/RevealLayer/blob/main/models/custom_model_xvae.py
https://github.com/360CVGroup/RevealLayer/blob/main/models/custom_pipeline.py
