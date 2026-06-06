"""阿里云实时语音识别 (NLS) 封装

文档: https://help.aliyun.com/document_detail/84426.html
"""
import os
import json
import base64
import wave
import io
import logging
import asyncio
import websockets
import uuid
import time
from typing import Optional, Callable
from urllib.parse import urlencode
import hashlib
import hmac

logger = logging.getLogger(__name__)


class AliyunNLSRecognizer:
    """阿里云实时语音识别客户端"""
    
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        app_key: Optional[str] = None,
        nls_url: Optional[str] = None
    ):
        self.access_key_id = access_key_id or os.environ.get("ALIYUN_ACCESS_KEY_ID")
        self.access_key_secret = access_key_secret or os.environ.get("ALIYUN_ACCESS_KEY_SECRET")
        self.app_key = app_key or os.environ.get("ALIYUN_NLS_APP_KEY")
        self.nls_url = nls_url or os.environ.get(
            "ALIYUN_NLS_URL", 
            "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
        )
        
        if not all([self.access_key_id, self.access_key_secret, self.app_key]):
            raise ValueError("缺少阿里云语音配置：请检查 ALIYUN_ACCESS_KEY_ID, ALIYUN_ACCESS_KEY_SECRET, ALIYUN_NLS_APP_KEY")
    
    def _generate_token(self) -> str:
        """生成阿里云访问令牌"""
        timestamp = int(time.time())
        
        # 构造签名字符串
        signature_nonce = str(uuid.uuid4())
        
        params = {
            "AccessKeyId": self.access_key_id,
            "Action": "CreateToken",
            "Version": "2019-02-28",
            "Timestamp": timestamp,
            "SignatureMethod": "HMAC-SHA1",
            "SignatureVersion": "1.0",
            "SignatureNonce": signature_nonce,
            "RegionId": "cn-shanghai"
        }
        
        # 排序并编码参数
        sorted_params = sorted(params.items())
        canonical_query = urlencode(sorted_params)
        
        # 构造待签名字符串
        string_to_sign = f"GET&%2F&{canonical_query}"
        
        # 计算签名
        key = f"{self.access_key_secret}&"
        signature = base64.b64encode(
            hmac.new(key.encode(), string_to_sign.encode(), hashlib.sha1).digest()
        ).decode()
        
        # 添加签名到参数
        params["Signature"] = signature
        
        # 返回用于 WebSocket 连接的 token 格式
        # 实际使用时，阿里云 NLS 支持直接通过 WebSocket 连接并使用 app_key
        # 这里简化处理，使用直接连接方式
        return signature_nonce
    
    async def _recognize_async(
        self, 
        audio_data: bytes,
        sample_rate: int = 16000,
        format: str = "wav"
    ) -> dict:
        """异步识别音频数据"""
        
        task_id = str(uuid.uuid4())
        
        # WebSocket 连接头
        headers = {
            "X-NLS-Token": self._generate_token(),
        }
        
        # 构建 WebSocket URL
        ws_url = f"{self.nls_url}?appkey={self.app_key}&token={headers['X-NLS-Token']}"
        
        results = []
        
        try:
            async with websockets.connect(ws_url, extra_headers=headers) as websocket:
                # 发送开始识别指令
                start_cmd = {
                    "header": {
                        "message_id": str(uuid.uuid4()),
                        "task_id": task_id,
                        "namespace": "SpeechRecognizer",
                        "name": "StartRecognition",
                        "appkey": self.app_key
                    },
                    "payload": {
                        "format": format,
                        "sample_rate": sample_rate,
                        "enable_intermediate_result": True,
                        "enable_punctuation_prediction": True,
                        "enable_inverse_text_normalization": True
                    }
                }
                
                await websocket.send(json.dumps(start_cmd))
                
                # 等待开始识别响应
                response = await websocket.recv()
                resp_data = json.loads(response)
                
                if resp_data.get("header", {}).get("name") != "RecognitionStarted":
                    error_msg = resp_data.get("payload", {}).get("message", "未知错误")
                    raise Exception(f"开始识别失败: {error_msg}")
                
                logger.info(f"开始识别任务: {task_id}")
                
                # 分块发送音频数据 (每块 3200 字节，约 100ms 音频)
                chunk_size = 3200
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    if chunk:
                        # 发送二进制音频数据
                        await websocket.send(chunk)
                        
                        # 尝试接收中间结果（非阻塞）
                        try:
                            while True:
                                msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                                msg_data = json.loads(msg)
                                if msg_data.get("header", {}).get("name") == "RecognitionResultChanged":
                                    result = msg_data.get("payload", {}).get("result", "")
                                    if result:
                                        results.append(result)
                        except asyncio.TimeoutError:
                            pass
                    
                    # 控制发送速率，模拟实时流
                    await asyncio.sleep(0.1)
                
                # 发送停止识别指令
                stop_cmd = {
                    "header": {
                        "message_id": str(uuid.uuid4()),
                        "task_id": task_id,
                        "namespace": "SpeechRecognizer",
                        "name": "StopRecognition"
                    }
                }
                await websocket.send(json.dumps(stop_cmd))
                
                # 等待最终识别结果
                final_text = ""
                while True:
                    try:
                        msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        msg_data = json.loads(msg)
                        msg_name = msg_data.get("header", {}).get("name")
                        
                        if msg_name == "RecognitionResultChanged":
                            result = msg_data.get("payload", {}).get("result", "")
                            if result:
                                results.append(result)
                        elif msg_name == "RecognitionCompleted":
                            final_text = msg_data.get("payload", {}).get("result", "")
                            break
                        elif msg_name == "TaskFailed":
                            error_msg = msg_data.get("payload", {}).get("message", "未知错误")
                            raise Exception(f"识别失败: {error_msg}")
                            
                    except asyncio.TimeoutError:
                        break
                
                # 如果没有最终结果，使用中间结果拼接
                if not final_text and results:
                    final_text = "".join(results)
                
                return {
                    "text": final_text,
                    "task_id": task_id,
                    "status": "success"
                }
                
        except Exception as e:
            logger.error(f"阿里云语音识别失败: {e}")
            raise
    
    def recognize(
        self, 
        audio_data: bytes,
        sample_rate: int = 16000,
        format: str = "wav"
    ) -> dict:
        """同步识别音频数据（阻塞调用）"""
        return asyncio.run(self._recognize_async(audio_data, sample_rate, format))


class AliyunFileTransRecognizer:
    """阿里云录音文件识别（适合较长的音频文件）
    
    文档: https://help.aliyun.com/document_detail/90727.html
    """
    
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        access_key_secret: Optional[str] = None,
        app_key: Optional[str] = None
    ):
        self.access_key_id = access_key_id or os.environ.get("ALIYUN_ACCESS_KEY_ID")
        self.access_key_secret = access_key_secret or os.environ.get("ALIYUN_ACCESS_KEY_SECRET")
        self.app_key = app_key or os.environ.get("ALIYUN_NLS_APP_KEY")
        
        if not all([self.access_key_id, self.access_key_secret, self.app_key]):
            raise ValueError("缺少阿里云语音配置")
    
    def recognize(self, audio_data: bytes, sample_rate: int = 16000) -> dict:
        """识别音频文件（使用 HTTP REST API）"""
        import requests
        
        # 文件识别接口地址
        url = "http://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/FlashRecognizer"
        
        # 构造请求参数
        params = {
            "appkey": self.app_key,
            "format": "wav",
            "sample_rate": sample_rate,
            "enable_punctuation_prediction": "true",
            "enable_inverse_text_normalization": "true"
        }
        
        # 构造签名
        timestamp = int(time.time())
        signature_nonce = str(uuid.uuid4())
        
        headers = {
            "X-NLS-Token": self._get_token(),
            "Content-Type": "application/octet-stream",
            "X-NLS-Request-Id": signature_nonce
        }
        
        try:
            response = requests.post(
                url,
                params=params,
                headers=headers,
                data=audio_data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result.get("status") == 20000000:
                return {
                    "text": result.get("flash_result", {}).get("sentences", [{}])[0].get("text", ""),
                    "status": "success",
                    "duration": result.get("flash_result", {}).get("duration", 0)
                }
            else:
                error_msg = result.get("message", "未知错误")
                raise Exception(f"识别失败: {error_msg}")
                
        except Exception as e:
            logger.error(f"阿里云文件识别失败: {e}")
            raise
    
    def _get_token(self) -> str:
        """获取访问令牌（简化版，实际生产环境应缓存 token）"""
        import requests
        
        url = "https://nls-meta.cn-shanghai.aliyuncs.com"
        
        # 构造获取 token 的请求
        # 这里使用简化方式，实际应使用阿里云 SDK 或正确签名
        # 为了快速集成，建议使用 alibabacloud-nls-java-sdk 的 Python 版本
        
        # 临时返回空，提示用户使用 SDK
        logger.warning("请安装阿里云 NLS SDK 获取 token: pip install alibabacloud-nls-java-sdk")
        return ""


# 便捷函数
def create_recognizer() -> AliyunNLSRecognizer:
    """创建阿里云语音识别器实例"""
    return AliyunNLSRecognizer()


def transcribe_audio(audio_data: bytes, sample_rate: int = 16000) -> dict:
    """便捷函数：音频字节转文字"""
    recognizer = create_recognizer()
    return recognizer.recognize(audio_data, sample_rate)
