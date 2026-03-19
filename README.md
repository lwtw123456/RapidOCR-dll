# 🚀 RapidOCR-dll

> 轻量级 OCR DLL 封装，基于最新 RapidOCR 核心实现

由于 [RapidOcrOnnx](https://github.com/RapidAI/RapidOcrOnnx) 长期未更新，本项目基于 [RapidOCR](https://github.com/RapidAI/RapidOCR) 最新版本（v3.7.0）  
将核心逻辑重写为 **C++ 实现**，并在此基础上进行了性能优化。

项目以 **DLL 形式提供统一接口**，通过 JSON 进行参数输入与结果输出。

当前已适配：

- ✅ PP-OCRv5 模型  
- ✅ ONNXRuntime 1.24.3  
- ✅ OpenCV 4.13.0  

## ✨ 特性

- 📦 **DLL 封装，跨语言调用友好**（C/C++/C#/Python 等）
- ⚡ **极致轻量**
  - DLL 仅 **3.77MB**
  - 模型约 **21MB**
- 🧠 **内存占用低**（峰值约 **350MB**）
- 🏎 **识别速度更快**（算法优化）
- ⚙️ **灵活配置**（JSON 参数控制）
- 🧩 **模块解耦，易于扩展**

------------------------------------------------------------------------

## 📁 项目结构

    .
    ├── include/              # 头文件
    ├── src/                  # 源码实现
    ├── third_party/          # 第三方依赖
	├── models/               # OCR模型（运行时需要）
	├── python/               # Python 封装
	├── build_release.bat     # 一键构建脚本
    ├── CMakeLists.txt
	└── README.md

------------------------------------------------------------------------

## 🚀 一键构建

``` bash
build_release.bat
```

### 功能

-   编译 Release 版本
-   生成 `RapidOCR.dll`
-   若存在 UPX，则自动压缩

------------------------------------------------------------------------

## 📦 使用说明

无需自行编译，你可以直接从 **Releases 页面下载 DLL 即可使用**。

### 获取方式

1. 打开项目的 **Releases 页面**
2. 下载最新版本中的：

RapidOCR.dll

---

### 最小使用步骤

1. 准备目录结构：

your_app/

├── RapidOCR.dll

├── models/

    ├── ch_PP-OCRv5_mobile_det.onnx
	
    ├── ch_ppocr_mobile_v2.0_cls_infer.onnx
	
    └── ch_PP-OCRv5_rec_mobile_infer.onnx

2. 在你的程序中加载 DLL，并调用接口：

- RapidOcrFromPathW（文件路径）
- RapidOcrFromBytes（内存数据）

### 调用流程

加载 DLL
  ↓
调用 OCR 接口
  ↓
获取 JSON 结果
  ↓
（可选）调用 RapidOcrJsonGetLineText 获取纯文本

### 注意事项

- 默认会在 DLL 同级目录查找 models/
- optionsJson 需为 UTF-8 编码
- 图片路径接口使用 wchar_t
- 返回 JSON 为 UTF-8 字符串
- 调用示例见： python/python_api.py

------------------------------------------------------------------------

### 接口说明

#### `RapidOcrFromPathW`

通过图片文件路径执行 OCR。

参数：

- `imagePath`：宽字符图片路径
- `optionsJson`：UTF-8 编码 JSON 配置，可传 `NULL`

返回：

- `const char*`，UTF-8 JSON 字符串

#### `RapidOcrFromBytes`

通过内存中的图片字节执行 OCR。

参数：

- `imageBytes`：图片字节数据
- `imageBytesLength`：字节长度
- `optionsJson`：UTF-8 编码 JSON 配置，可传 `NULL`

返回：

- `const char*`，UTF-8 JSON 字符串

#### `RapidOcrJsonGetLineText`

从 OCR 返回的 JSON 中提取纯文本，并按行拼接。

参数：

- `jsonBytes`：OCR 返回 JSON
- `jsonBytesLength`：JSON 字节长度
- `outTextLength`：输出文本长度

返回：

- `const unsigned char*`，UTF-8 文本字节

---

## optionsJson 参数说明

所有参数均为可选项。未传入时使用默认值。

### 示例

```json
{
  "model_dir": "models",
  "use_cls": true,
  "max_side_len": 2000,
  "min_side_len": 30,
  "limit_side_len": 736.0,
  "limit_type": "min",
  "thresh": 0.3,
  "box_thresh": 0.5,
  "max_candidates": 1000,
  "unclip_ratio": 1.6,
  "use_dilation": true,
  "score_mode": "fast",
  "merge_code_lines": false
}
```

### 参数详解

#### 基础配置

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `model_dir` | `string` | 模型目录路径 | `"models"` |
| `use_cls` | `bool` | 是否启用方向分类 | `true` |

---

#### 尺寸控制

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `max_side_len` | `int` | 输入图像最大边长限制 | `2000` |
| `min_side_len` | `int` | 输入图像最小边长限制 | `30` |

用于控制推理性能与精度。

---

#### 检测相关（Detector）

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `limit_side_len` | `float` | 检测缩放基准 | `736.0` |
| `limit_type` | `string` | `"min"` 或 `"max"` | `"min"` |
| `thresh` | `float` | 前景阈值 | `0.3` |
| `box_thresh` | `float` | 文本框过滤阈值 | `0.5` |
| `max_candidates` | `int` | 最大候选框数量 | `1000` |

---

#### 文本框处理

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `unclip_ratio` | `float` | 文本框扩展比例 | `1.6` |
| `use_dilation` | `bool` | 是否使用膨胀操作 | `true` |
| `score_mode` | `string` | `"fast"` 或 `"slow"` | `"fast"` |

---

#### 特殊优化

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `merge_code_lines` | `bool` | 合并代码行，适用于代码截图 | `false` |

该参数会影响检测框合并逻辑，适合代码编辑器截图、终端截图等文本场景。

------------------------------------------------------------------------

## 返回数据结构

接口成功或失败都会返回 JSON 字符串。

### 成功示例

```json
{
  "code": 100,
  "message": "ok",
  "data": [
    {
      "box": [[12, 34], [220, 34], [220, 68], [12, 68]],
      "score": 0.98765,
      "text": "Hello World"
    }
  ]
}
```

### 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | `int` | 状态码 |
| `message` | `string` | 状态信息 |
| `data` | `array` | OCR 结果数组 |

### `data` 数组元素结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `box` | `array` | 文本框四点坐标，顺序为四边形四个顶点 |
| `score` | `float` | 置信度 |
| `text` | `string` | 识别文本 |

### `box` 字段说明

```json
"box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
```

表示文本框四个顶点坐标。

### `score` 字段说明

- 当存在字符级置信度时，返回字符分数平均值
- 否则退化为文本框分数
- 保留到 5 位小数

------------------------------------------------------------------------

## 状态码

| 状态码 | 含义 |
|--------|------|
| `100` | 成功 |
| `400` | 请求参数错误 |
| `404` | 文件不存在或读取失败 |
| `500` | 初始化失败 |
| `501` | 内部错误 |

### 失败示例

```json
{
  "code": 404,
  "message": "failed to read image file",
  "data": []
}
```
------------------------------------------------------------------------

## 📄 License

MIT License

------------------------------------------------------------------------

## ⭐ 支持

如果对你有帮助，欢迎 Star ⭐
