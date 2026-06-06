"""语音识别模块 - 支持阿里云实时识别和本地Whisper离线识别"""
import os
import io
import wave
import tempfile
import logging

logger = logging.getLogger(__name__)

# 设置HuggingFace镜像源（中国大陆）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 模型配置：CPU int8模式，平衡速度与精度
# 可选模型：tiny(75MB) / base(150MB) / small(500MB) / medium(1.5GB)
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

# 是否使用阿里云语音识别
USE_ALIYUN_ASR = os.environ.get("USE_ALIYUN_ASR", "true").lower() == "true"


class WhisperRecognizer:
    """Whisper语音识别器（单例，延迟加载）"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_size: str = None):
        if self._model is not None:
            return
        size = model_size or MODEL_SIZE
        logger.info(f"加载Whisper模型: {size} (CPU int8)...")
        from faster_whisper import WhisperModel
        self._model = WhisperModel(size, device="cpu", compute_type="int8")
        logger.info(f"Whisper模型加载完成: {size}")

    def transcribe(self, audio_data: bytes, language: str = "zh") -> dict:
        """将音频字节数据转换为文字
        
        Args:
            audio_data: WAV音频字节数据
            language: 语言代码，默认中文
            
        Returns:
            {"text": "识别文字", "segments": [...], "duration": 秒}
        """
        self.load_model()
        audio_file = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                audio_file = f.name

            segments, info = self._model.transcribe(
                audio_file,
                language=language,
                beam_size=3,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            segments_list = list(segments)
            text = "".join(seg.text for seg in segments_list)

            result = {
                "text": text.strip(),
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in segments_list
                ],
                "duration": info.duration if info.duration else 0,
                "language": info.language,
            }
            return result

        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            raise
        finally:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)


class AliyunASRAdapter:
    """阿里云语音识别适配器"""
    _instance = None
    _recognizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_recognizer(self):
        if self._recognizer is None:
            from .aliyun_asr import AliyunNLSRecognizer
            self._recognizer = AliyunNLSRecognizer()
        return self._recognizer
    
    def transcribe(self, audio_data: bytes, language: str = "zh") -> dict:
        """使用阿里云语音识别"""
        recognizer = self._get_recognizer()
        
        # 阿里云实时语音识别
        result = recognizer.recognize(audio_data, sample_rate=16000)
        
        return {
            "text": result.get("text", ""),
            "segments": [],
            "duration": 0,
            "language": language,
            "task_id": result.get("task_id", ""),
            "source": "aliyun"
        }


# 全局单例
_whisper_recognizer = None
_aliyun_adapter = None


def get_whisper_recognizer() -> WhisperRecognizer:
    global _whisper_recognizer
    if _whisper_recognizer is None:
        _whisper_recognizer = WhisperRecognizer()
    return _whisper_recognizer


def get_aliyun_recognizer() -> AliyunASRAdapter:
    global _aliyun_adapter
    if _aliyun_adapter is None:
        _aliyun_adapter = AliyunASRAdapter()
    return _aliyun_adapter


def transcribe_audio(audio_data: bytes, language: str = "zh") -> dict:
    """便捷函数：音频→文字
    
    根据 USE_ALIYUN_ASR 环境变量自动选择识别引擎
    """
    if USE_ALIYUN_ASR:
        try:
            logger.info("使用阿里云语音识别...")
            recognizer = get_aliyun_recognizer()
            return recognizer.transcribe(audio_data, language)
        except Exception as e:
            logger.warning(f"阿里云识别失败，回退到本地Whisper: {e}")
            # 如果阿里云失败，回退到本地Whisper
            recognizer = get_whisper_recognizer()
            result = recognizer.transcribe(audio_data, language)
            result["source"] = "whisper"
            result["fallback"] = True
            return result
    else:
        logger.info("使用本地Whisper识别...")
        recognizer = get_whisper_recognizer()
        result = recognizer.transcribe(audio_data, language)
        result["source"] = "whisper"
        return result


def transcribe_file(file_path: str, language: str = "zh") -> dict:
    """便捷函数：音频文件→文字"""
    with open(file_path, "rb") as f:
        audio_data = f.read()
    return transcribe_audio(audio_data, language)


# 向后兼容的别名
get_recognizer = get_whisper_recognizer
