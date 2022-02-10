## 素材来源

`icon` 来自[企鹅物流前端](https://github.com/penguin-statistics/frontend-v2), 运行 `images/download-icons.sh` 下载

`item` 来自[明日方舟工具箱](https://github.com/arkntools/arknights-toolbox), 运行 `images/download-items.sh` 下载

`collect` 来自[prts.wiki](http://prts.wiki/w/%E9%81%93%E5%85%B7%E4%B8%80%E8%A7%88), 运行 `dl_data.py` 下载


## 训练

### 使用 collect 中的素材进行训练

运行 `train_torch_from_collect.py` 中的 `train()` 即可, 生成模型为 `model.pth` 与 `ark_material.onnx`

## 使用

依赖安装:

```bash
# 物品识别可以直接使用 opencv 内置的 dnn 模块
# 数量识别需要 PaddleOCR 或其它具有 std 功能的 ocr 引擎
pip install opencv-python, numpy, ppocr-onnx
```

需配合 `ark_material.onnx` 及 `index_itemid_relation.json` 使用.

具体使用方式见 [demo.py](./demo.py)
