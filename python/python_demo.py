from python_api import OcrEngine
import time

engine = OcrEngine()

if engine.available:
    # 1. 传路径
    tmp = time.time()
    result = engine.ocr("Screenshot.png")
    print(time.time() - tmp)