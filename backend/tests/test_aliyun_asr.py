"""测试阿里云语音识别配置"""
import os
import sys

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("阿里云语音识别配置检查")
print("=" * 60)

# 检查配置
access_key_id = os.environ.get("ALIYUN_ACCESS_KEY_ID")
access_key_secret = os.environ.get("ALIYUN_ACCESS_KEY_SECRET")
app_key = os.environ.get("ALIYUN_NLS_APP_KEY")
use_aliyun = os.environ.get("USE_ALIYUN_ASR", "true")

print(f"\n[OK] AccessKey ID: {'已配置' if access_key_id else '未配置'}")
print(f"[OK] AccessKey Secret: {'已配置' if access_key_secret else '未配置'}")
print(f"[OK] AppKey: {'已配置' if app_key else '未配置'}")
print(f"[OK] 使用阿里云ASR: {use_aliyun}")

if access_key_id:
    print(f"  ID 前缀: {access_key_id[:8]}...")

print("\n" + "=" * 60)
print("检查依赖")
print("=" * 60)

try:
    import websockets
    print("[OK] websockets 已安装")
except ImportError:
    print("[X] websockets 未安装，请运行: pip install websockets")

try:
    import requests
    print("[OK] requests 已安装")
except ImportError:
    print("[X] requests 未安装，请运行: pip install requests")

print("\n" + "=" * 60)
print("测试导入")
print("=" * 60)

try:
    from perception.aliyun_asr import AliyunNLSRecognizer
    print("[OK] AliyunNLSRecognizer 导入成功")
    
    # 尝试实例化
    if all([access_key_id, access_key_secret, app_key]):
        recognizer = AliyunNLSRecognizer()
        print("[OK] AliyunNLSRecognizer 实例化成功")
    else:
        print("[!] 缺少配置，跳过实例化测试")
        
except Exception as e:
    print(f"[X] 导入失败: {e}")

print("\n" + "=" * 60)
print("下一步操作")
print("=" * 60)
print("""
1. 请在阿里云控制台获取 AppKey:
   https://nls-portal.console.aliyun.com/applist
   
2. 将 AppKey 填入 .env 文件:
   ALIYUN_NLS_APP_KEY=your_actual_app_key
   
3. 安装依赖:
   pip install websockets requests
   
4. 启动后端服务测试:
   cd backend
   python -m uvicorn api.main:app --reload
   
5. 打开前端页面测试语音输入功能
""")

print("=" * 60)
