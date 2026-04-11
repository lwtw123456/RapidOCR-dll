from python_api import OcrEngine
import os

engine = OcrEngine()

if not engine.available:
    print("OCR引擎不可用，请检查RapidOCR.dll是否存在")
    exit(1)

print("OCR引擎加载成功")

# 1. 传入图片路径
result = engine.ocr("Screenshot.png")
if result:
    print("从路径识别成功:")
    print(result)
else:
    print("从路径识别失败")

# 2. 传入图片字节数据
with open("Screenshot.png", "rb") as f:
    image_bytes = f.read()

result = engine.ocr(image_bytes)  # 传入bytes
if result:
    print("传入图片字节数据识别成功:")
    print(result)
else:
    print("传入图片字节数据识别失败")


# 3. 带选项的识别
options = {
    "use_cls": False,   # 不使用方向分类
    "only_text": True,  # 只获取纯文本   
}

result = engine.ocr("Screenshot.png", options=options)
if result:
    print(result)
    print("带选项识别完成")
