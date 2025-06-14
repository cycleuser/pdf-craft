# -*- coding: utf-8 -*-
"""
多OCR引擎管理器
支持多种高性能OCR引擎，提供统一接口
"""

import os
import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io

logger = logging.getLogger(__name__)

class OCREngine:
    """OCR引擎基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.available = False
        self.error_message = ""
        
    def check_availability(self) -> bool:
        """检查引擎是否可用"""
        return self.available
        
    def extract_text_from_image(self, image: Image.Image, language: str = 'auto') -> str:
        """从图像提取文本"""
        raise NotImplementedError
        
    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int, language: str = 'auto') -> Dict[str, Any]:
        """从PDF页面提取文本"""
        raise NotImplementedError

class PDFCraftEngine(OCREngine):
    """PDF-Craft引擎"""
    
    def __init__(self, model_dir: str, device: str = 'cpu'):
        super().__init__("PDF-Craft", "原始PDF-Craft引擎，支持复杂文档结构")
        self.model_dir = model_dir
        self.device = device
        self.extractor = None
        self._check_availability()
        
    def _check_availability(self):
        try:
            from pdf_craft import PDFPageExtractor
            self.extractor = PDFPageExtractor(
                device=self.device,
                model_dir_path=self.model_dir
            )
            self.available = True
            logger.info("PDF-Craft引擎初始化成功")
        except Exception as e:
            self.available = False
            self.error_message = str(e)
            logger.warning(f"PDF-Craft引擎不可用: {e}")
            
    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int, language: str = 'auto') -> Dict[str, Any]:
        if not self.available:
            return {"text": "", "error": self.error_message}
            
        try:
            # 使用PDF-Craft处理单页
            results = list(self.extractor.extract(pdf=pdf_path))
            if page_num < len(results):
                return {
                    "text": str(results[page_num]),
                    "confidence": 0.9,
                    "processing_time": 0,
                    "engine": self.name
                }
            return {"text": "", "error": "页面不存在"}
        except Exception as e:
            return {"text": "", "error": str(e)}

class TesseractEngine(OCREngine):
    """Tesseract OCR引擎"""
    
    def __init__(self):
        super().__init__("Tesseract", "Google Tesseract OCR，支持100+语言")
        self._check_availability()
        
    def _check_availability(self):
        try:
            import pytesseract
            from PIL import Image
            # 测试Tesseract是否可用
            test_image = Image.new('RGB', (100, 50), color='white')
            pytesseract.image_to_string(test_image)
            self.available = True
            logger.info("Tesseract引擎初始化成功")
        except Exception as e:
            self.available = False
            self.error_message = str(e)
            logger.warning(f"Tesseract引擎不可用: {e}")
            
    def extract_text_from_image(self, image: Image.Image, language: str = 'auto') -> str:
        if not self.available:
            return ""
            
        try:
            import pytesseract
            
            # 语言映射
            lang_map = {
                'auto': 'chi_sim+chi_tra+eng',
                'chinese': 'chi_sim+chi_tra',
                'english': 'eng',
                'japanese': 'jpn',
                'korean': 'kor',
                'french': 'fra',
                'german': 'deu',
                'spanish': 'spa',
                'russian': 'rus',
                'arabic': 'ara'
            }
            
            tesseract_lang = lang_map.get(language, 'chi_sim+chi_tra+eng')
            
            # 配置Tesseract参数
            config = '--oem 3 --psm 6'
            
            text = pytesseract.image_to_string(image, lang=tesseract_lang, config=config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR失败: {e}")
            return ""
            
    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int, language: str = 'auto') -> Dict[str, Any]:
        if not self.available:
            return {"text": "", "error": self.error_message}
            
        start_time = time.time()
        try:
            # 将PDF页面转换为图像
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                return {"text": "", "error": "页面不存在"}
                
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)  # 提高分辨率
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            doc.close()
            
            # OCR识别
            text = self.extract_text_from_image(image, language)
            processing_time = time.time() - start_time
            
            return {
                "text": text,
                "confidence": 0.8,
                "processing_time": processing_time,
                "engine": self.name
            }
        except Exception as e:
            return {"text": "", "error": str(e)}

class EasyOCREngine(OCREngine):
    """EasyOCR引擎"""
    
    def __init__(self, gpu: bool = False):
        super().__init__("EasyOCR", "高精度深度学习OCR，支持80+语言")
        self.gpu = gpu
        self.reader = None
        self._check_availability()
        
    def _check_availability(self):
        try:
            import easyocr
            # 初始化EasyOCR
            self.reader = easyocr.Reader(['ch_sim', 'ch_tra', 'en'], gpu=self.gpu)
            self.available = True
            logger.info(f"EasyOCR引擎初始化成功 (GPU: {self.gpu})")
        except Exception as e:
            self.available = False
            self.error_message = str(e)
            logger.warning(f"EasyOCR引擎不可用: {e}")
            
    def extract_text_from_image(self, image: Image.Image, language: str = 'auto') -> str:
        if not self.available:
            return ""
            
        try:
            # 转换PIL图像为numpy数组
            img_array = np.array(image)
            
            # EasyOCR识别
            results = self.reader.readtext(img_array)
            
            # 提取文本
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # 置信度阈值
                    texts.append(text)
                    
            return '\n'.join(texts)
        except Exception as e:
            logger.error(f"EasyOCR识别失败: {e}")
            return ""
            
    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int, language: str = 'auto') -> Dict[str, Any]:
        if not self.available:
            return {"text": "", "error": self.error_message}
            
        start_time = time.time()
        try:
            # 将PDF页面转换为图像
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                return {"text": "", "error": "页面不存在"}
                
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            doc.close()
            
            # OCR识别
            text = self.extract_text_from_image(image, language)
            processing_time = time.time() - start_time
            
            return {
                "text": text,
                "confidence": 0.85,
                "processing_time": processing_time,
                "engine": self.name
            }
        except Exception as e:
            return {"text": "", "error": str(e)}

class PaddleOCREngine(OCREngine):
    """PaddleOCR引擎"""
    
    def __init__(self, gpu: bool = False):
        super().__init__("PaddleOCR", "百度PaddleOCR，中文识别效果优秀")
        self.gpu = gpu
        self.ocr = None
        self._check_availability()
        
    def _check_availability(self):
        try:
            from paddleocr import PaddleOCR
            # 初始化PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=self.gpu,
                show_log=False
            )
            self.available = True
            logger.info(f"PaddleOCR引擎初始化成功 (GPU: {self.gpu})")
        except Exception as e:
            self.available = False
            self.error_message = str(e)
            logger.warning(f"PaddleOCR引擎不可用: {e}")
            
    def extract_text_from_image(self, image: Image.Image, language: str = 'auto') -> str:
        if not self.available:
            return ""
            
        try:
            # 转换PIL图像为numpy数组
            img_array = np.array(image)
            
            # PaddleOCR识别
            results = self.ocr.ocr(img_array, cls=True)
            
            # 提取文本
            texts = []
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        if confidence > 0.5:
                            texts.append(text)
                            
            return '\n'.join(texts)
        except Exception as e:
            logger.error(f"PaddleOCR识别失败: {e}")
            return ""
            
    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int, language: str = 'auto') -> Dict[str, Any]:
        if not self.available:
            return {"text": "", "error": self.error_message}
            
        start_time = time.time()
        try:
            # 将PDF页面转换为图像
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                return {"text": "", "error": "页面不存在"}
                
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            doc.close()
            
            # OCR识别
            text = self.extract_text_from_image(image, language)
            processing_time = time.time() - start_time
            
            return {
                "text": text,
                "confidence": 0.9,
                "processing_time": processing_time,
                "engine": self.name
            }
        except Exception as e:
            return {"text": "", "error": str(e)}

class RapidOCREngine(OCREngine):
    """RapidOCR引擎"""
    
    def __init__(self):
        super().__init__("RapidOCR", "轻量级高速OCR，基于ONNX Runtime")
        self.ocr = None
        self._check_availability()
        
    def _check_availability(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
            self.ocr = RapidOCR()
            self.available = True
            logger.info("RapidOCR引擎初始化成功")
        except Exception as e:
            self.available = False
            self.error_message = str(e)
            logger.warning(f"RapidOCR引擎不可用: {e}")
            
    def extract_text_from_image(self, image: Image.Image, language: str = 'auto') -> str:
        if not self.available:
            return ""
            
        try:
            # 转换PIL图像为numpy数组
            img_array = np.array(image)
            
            # RapidOCR识别
            result, elapse = self.ocr(img_array)
            
            # 提取文本
            texts = []
            if result:
                for line in result:
                    if len(line) >= 2:
                        text = line[1]
                        confidence = line[2] if len(line) > 2 else 1.0
                        if confidence > 0.5:
                            texts.append(text)
                            
            return '\n'.join(texts)
        except Exception as e:
            logger.error(f"RapidOCR识别失败: {e}")
            return ""
            
    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int, language: str = 'auto') -> Dict[str, Any]:
        if not self.available:
            return {"text": "", "error": self.error_message}
            
        start_time = time.time()
        try:
            # 将PDF页面转换为图像
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                return {"text": "", "error": "页面不存在"}
                
            page = doc[page_num]
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            doc.close()
            
            # OCR识别
            text = self.extract_text_from_image(image, language)
            processing_time = time.time() - start_time
            
            return {
                "text": text,
                "confidence": 0.85,
                "processing_time": processing_time,
                "engine": self.name
            }
        except Exception as e:
            return {"text": "", "error": str(e)}

class OCREngineManager:
    """OCR引擎管理器"""
    
    def __init__(self, model_dir: str, use_gpu: bool = False):
        self.model_dir = model_dir
        self.use_gpu = use_gpu
        self.engines = {}
        self._initialize_engines()
        
    def _initialize_engines(self):
        """初始化所有OCR引擎"""
        logger.info("初始化OCR引擎...")
        
        # PDF-Craft引擎
        device = 'cuda' if self.use_gpu else 'cpu'
        self.engines['pdf_craft'] = PDFCraftEngine(self.model_dir, device)
        
        # Tesseract引擎
        self.engines['tesseract'] = TesseractEngine()
        
        # EasyOCR引擎
        self.engines['easyocr'] = EasyOCREngine(gpu=self.use_gpu)
        
        # PaddleOCR引擎
        self.engines['paddleocr'] = PaddleOCREngine(gpu=self.use_gpu)
        
        # RapidOCR引擎
        self.engines['rapidocr'] = RapidOCREngine()
        
        # 统计可用引擎
        available_engines = [name for name, engine in self.engines.items() if engine.available]
        logger.info(f"可用OCR引擎: {available_engines}")
        
    def get_available_engines(self) -> Dict[str, Dict[str, Any]]:
        """获取可用的OCR引擎列表"""
        result = {}
        for name, engine in self.engines.items():
            result[name] = {
                'name': engine.name,
                'description': engine.description,
                'available': engine.available,
                'error_message': engine.error_message if not engine.available else ""
            }
        return result
        
    def extract_text_from_pdf(self, pdf_path: str, engine_name: str = 'pdf_craft', 
                             language: str = 'auto', progress_callback=None) -> List[Dict[str, Any]]:
        """使用指定引擎从PDF提取文本"""
        if engine_name not in self.engines:
            raise ValueError(f"未知的OCR引擎: {engine_name}")
            
        engine = self.engines[engine_name]
        if not engine.available:
            raise RuntimeError(f"OCR引擎不可用: {engine.error_message}")
            
        # 获取PDF页数
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        results = []
        for page_num in range(total_pages):
            if progress_callback:
                progress_callback(page_num + 1, total_pages)
                
            result = engine.extract_text_from_pdf_page(pdf_path, page_num, language)
            result['page_number'] = page_num + 1
            results.append(result)
            
        return results
        
    def benchmark_engines(self, test_pdf_path: str, max_pages: int = 3) -> Dict[str, Dict[str, Any]]:
        """对比不同OCR引擎的性能"""
        results = {}
        
        for engine_name, engine in self.engines.items():
            if not engine.available:
                results[engine_name] = {
                    'available': False,
                    'error': engine.error_message
                }
                continue
                
            try:
                start_time = time.time()
                
                # 测试前几页
                doc = fitz.open(test_pdf_path)
                test_pages = min(max_pages, len(doc))
                doc.close()
                
                total_text_length = 0
                page_times = []
                
                for page_num in range(test_pages):
                    page_start = time.time()
                    result = engine.extract_text_from_pdf_page(test_pdf_path, page_num)
                    page_time = time.time() - page_start
                    page_times.append(page_time)
                    
                    if 'text' in result:
                        total_text_length += len(result['text'])
                        
                total_time = time.time() - start_time
                avg_page_time = sum(page_times) / len(page_times) if page_times else 0
                
                results[engine_name] = {
                    'available': True,
                    'total_time': total_time,
                    'avg_page_time': avg_page_time,
                    'pages_per_second': 1.0 / avg_page_time if avg_page_time > 0 else 0,
                    'total_text_length': total_text_length,
                    'chars_per_second': total_text_length / total_time if total_time > 0 else 0,
                    'test_pages': test_pages
                }
                
            except Exception as e:
                results[engine_name] = {
                    'available': False,
                    'error': str(e)
                }
                
        return results 