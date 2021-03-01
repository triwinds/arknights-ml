## 素材来源

`icon` 来自[企鹅物流前端](https://github.com/penguin-statistics/frontend-v2), 运行 `images/download-icons.sh` 下载

`item` 来自[明日方舟工具箱](https://github.com/arkntools/arknights-toolbox), 运行 `images/download-items.sh` 下载

`collect` 来自截图收集, 运行 `train_torch_from_collect.py` 中的 `prepare_train_resource()` 进行收集

## 截图要求

需要 720p 分辨率的截图

## 训练

### 使用 icon 中的素材进行训练

 `train_torch_from_icon.py` , 直接执行 `train()` 即可. 使用 icon 中的素材进行训练, 生成模型为 `model2.bin`.

无法识别 icon 素材外的物品. 

### 使用 collect 中的素材进行训练

1. 准备素材

collect 素材整理格式: `collect/<itemId>/*.png`

 `train_torch_from_collect.py` , 使用 collect 中的素材 (基础材料的素材会在启动时从 icon 复制) 进行训练, collect/other 中的素材运行 `prepare_train_resource()` 收集.

>  `prepare_train_resource()` 从 adb 截图中进行收集, 需要手工根据控制台中的输出与展示的图像进行判断
>
> 在 opencv 图像的窗口中按回车键(keycode == 13)会将该素材放入相应 item id 的文件夹;
>
> 按其他按键会将该素材放入 other 文件夹.

2. 执行训练

直接执行 `train()` 即可, 生成模型为 `model3.bin`

## 使用

直接执行 `test()` 即可.

## 导出 onnx

1. 准备一张仓库的截图放到 `images/screen.png` 
2. 运行 `train_torch_from_collect.py` 中的 `export_onnx()`