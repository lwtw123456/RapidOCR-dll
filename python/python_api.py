import ctypes
import os
import json


class OcrEngine:
    def __init__(self, options=None):
        self._dll = None
        self._loaded = False
        self._options = options
        self._dll_path = os.path.abspath("RapidOCR.dll")

    @property
    def available(self):
        self._lazy_load()
        return self._dll is not None

    def _lazy_load(self):
        if self._loaded:
            return

        self._loaded = True

        if not os.path.exists(self._dll_path):
            return

        try:
            dll = ctypes.WinDLL(self._dll_path)

            dll.RapidOcrFromPathW.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_char_p,
            ]
            dll.RapidOcrFromPathW.restype = ctypes.c_char_p

            dll.RapidOcrFromBytes.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_int,
                ctypes.c_char_p,
            ]
            dll.RapidOcrFromBytes.restype = ctypes.c_char_p

            dll.RapidOcrJsonGetLineText.argtypes = [
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
            ]
            dll.RapidOcrJsonGetLineText.restype = ctypes.POINTER(ctypes.c_ubyte)

            self._dll = dll

        except Exception:
            self._dll = None

    def _merge_options(self, options=None):
        if self._options is None and options is None:
            return None

        base = {}
        if isinstance(self._options, dict):
            base.update(self._options)

        if isinstance(options, dict):
            base.update(options)

        return base if base else None

    def _options_json(self, options=None):
        final_options = self._merge_options(options)
        if not final_options:
            return None
        return json.dumps(final_options, ensure_ascii=False).encode("utf-8")

    def ocr(self, image, only_text=False, options=None):
        if image is None:
            return None

        if not self.available:
            return None

        options_json = self._options_json(options)

        try:
            if isinstance(image, str):
                json_bytes = self._dll.RapidOcrFromPathW(image, options_json)

            elif isinstance(image, bytes):
                if not image:
                    return None
                buf = (ctypes.c_ubyte * len(image)).from_buffer_copy(image)
                json_bytes = self._dll.RapidOcrFromBytes(
                    buf,
                    len(image),
                    options_json,
                )
            else:
                return None

            if not json_bytes:
                return None

            if only_text:
                length = ctypes.c_int(0)
                text_ptr = self._dll.RapidOcrJsonGetLineText(
                    json_bytes,
                    len(json_bytes),
                    ctypes.byref(length),
                )
                if not text_ptr or length.value <= 0:
                    return ""
                return ctypes.string_at(text_ptr, length.value).decode("utf-8", errors="ignore")

            return json.loads(json_bytes.decode("utf-8"))

        except Exception:
            return None