# 超声乳腺分类及 BI-RADS 特征检测（精简版）

这是一个精简后的 PyTorch 项目，包含 2 个统一入口：
- `train.py`：统一训练入口（`--task cls|feat`）
- `infer.py`：使用两个模型分别做分类与特征推理

## 项目结构

```text
.
├─datasets_core.py
├─train.py
├─infer.py
├─requirements.txt
├─.gitignore
├─LICENSE
└─data/                   # 本地数据，默认不上传
```

## 环境安装

推荐 Python `3.10+`

```bash
pip install -r requirements.txt
```

## 数据结构

### 分类数据

```text
data/乳腺分类训练数据集/
├─train/
│  ├─2类/images/
│  ├─3类/images/
│  ├─4A类/images/
│  ├─4B类/images/
│  ├─4C类/images/
│  └─5类/images/
└─test/
  └─(同上结构)
```

### 特征数据

```text
data/乳腺特征训练数据集/
├─train/
│  ├─images/
│  ├─boundary_labels/
│  ├─calcification_labels/
│  ├─direction_labels/
│  └─shape_labels/
└─test/
  └─(同上结构)
```

## 训练

```bash
python train.py --task cls --train-dir "data/乳腺分类训练数据集/train" --val-dir "data/乳腺分类训练数据集/test" --pretrained --epochs 20 --batch-size 16
python train.py --task feat --train-dir "data/乳腺特征训练数据集/train" --val-dir "data/乳腺特征训练数据集/test" --pretrained --epochs 20 --batch-size 16
```

常用参数：

```bash
python train.py --task cls --train-dir <分类训练目录> --val-dir <分类验证目录> --save-dir outputs
python train.py --task feat --train-dir <特征训练目录> --val-dir <特征验证目录> --save-dir outputs
```

## 推理

```bash
python infer.py --cls-weights "outputs/cls/best_cls_model.pth" --feat-weights "outputs/feat/best_feat_model.pth"
```

输出在 `outputs/predict/`：
- `cls_predictions.txt`
- `feat_predictions.txt`

## 上传到 GitHub

```bash
git init
git add .
git commit -m "refactor: simplify project structure"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

## 备注

- `data/`、权重、日志与输出目录已在 `.gitignore` 中排除。
- 训练脚本使用命令行参数，不再依赖硬编码绝对路径。
- 此项目所使用数据集为24年算法精英大赛超声乳腺数据集，只是为了提供一个思路，希望可以对你有所帮助
- 协作创作者：zqr5 & FeiYu Sun
