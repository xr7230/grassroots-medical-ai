"""
API服务主入口 - 重构版
采用分而治之策略：先提取、再校验、最后填模板
"""
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.medical_agent import MedicalRecordAgent

app = FastAPI(
    title="AI医疗文书智能体 - 重构版",
    description="采用分而治之策略：先提取、再校验、最后填模板",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = MedicalRecordAgent()

# 前端静态文件路径
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

# 历史记录存储文件
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "history.json")


class MedicalRecordRequest(BaseModel):
    input_text: str
    record_type: str = "admission_note"


class MedicalRecordResponse(BaseModel):
    status: str
    record: Optional[Dict] = None
    formatted_output: Optional[str] = None
    qc_info: Optional[Dict] = None
    message: Optional[str] = None
    disclaimer: Optional[str] = None


def load_history() -> List[Dict]:
    """加载历史记录"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(records: List[Dict]):
    """保存历史记录"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def add_to_history(record_id: str, record_type: str, content: str, input_text: str):
    """添加记录到历史"""
    records = load_history()
    records.insert(0, {
        "record_id": record_id,
        "record_type": record_type,
        "content": content,
        "input_text": input_text,
        "created_at": datetime.now().isoformat()
    })
    # 最多保存500条
    records = records[:500]
    save_history(records)


@app.get("/")
async def root():
    """返回前端页面"""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return HTMLResponse("<h1>前端文件未找到</h1>", status_code=404)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "3.0.0"}


@app.post("/api/v3/generate-record", response_model=MedicalRecordResponse)
async def generate_medical_record(request: MedicalRecordRequest):
    try:
        result = agent.process_input(request.input_text, request.record_type)
        
        if result["status"] == "awaiting_input":
            return MedicalRecordResponse(
                status="awaiting_input",
                message=result["message"]
            )
        
        # 保存到历史记录
        add_to_history(
            record_id=result["record"].record_id,
            record_type=request.record_type,
            content=result["formatted_output"],
            input_text=request.input_text
        )
        
        return MedicalRecordResponse(
            status="success",
            record={
                "record_id": result["record"].record_id,
                "chief_complaint": result["record"].chief_complaint,
                "present_illness": result["record"].present_illness,
                "physical_exam": result["record"].physical_exam,
                "auxiliary_exam": result["record"].auxiliary_exam,
                "preliminary_diagnosis": result["record"].preliminary_diagnosis
            },
            formatted_output=result["formatted_output"],
            qc_info=result.get("qc_info"),
            disclaimer=result.get("disclaimer")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v2/generate-record", response_model=MedicalRecordResponse)
async def generate_medical_record_v2(request: MedicalRecordRequest):
    """v2版本接口 - 兼容慢病随访"""
    return await generate_medical_record(request)


@app.post("/api/v2/generate-public-health-report")
async def generate_public_health_report(request: MedicalRecordRequest):
    """生成公卫报表"""
    try:
        result = agent.process_input(request.input_text, request.record_type)
        
        if result["status"] == "awaiting_input":
            return {"status": "awaiting_input", "message": result["message"]}
        
        # 保存到历史记录
        add_to_history(
            record_id=result["record"].record_id,
            record_type="public_health_report",
            content=result["formatted_output"],
            input_text=request.input_text
        )
        
        # 简化的公卫报表格式
        report = {
            "基本信息": {
                "姓名": result["record"].patient_info.split("姓名：")[1].split("\n")[0].strip() if "姓名：" in result["record"].patient_info else "待补充",
                "性别": result["record"].patient_info.split("性别：")[1].split("\n")[0].strip() if "性别：" in result["record"].patient_info else "待补充",
                "年龄": result["record"].patient_info.split("年龄：")[1].split("\n")[0].strip() if "年龄：" in result["record"].patient_info else "待补充",
            },
            "主要诊断": result["record"].preliminary_diagnosis[0] if result["record"].preliminary_diagnosis else "待补充",
            "随访记录": result["formatted_output"],
            "健康指导": "1. 规律服药，定期监测血压、血糖\n2. 合理饮食，适量运动\n3. 戒烟限酒，保持良好心态\n4. 定期复诊，如有不适及时就医"
        }
        
        return {
            "status": "success",
            "report": report,
            "formatted_output": result["formatted_output"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/history")
async def get_history(keyword: Optional[str] = None, record_type: Optional[str] = None, limit: int = 50):
    """获取历史记录列表"""
    try:
        records = load_history()
        
        # 按类型过滤
        if record_type:
            records = [r for r in records if r["record_type"] == record_type]
        
        # 按关键词搜索
        if keyword:
            keyword = keyword.lower()
            records = [r for r in records 
                      if keyword in r["content"].lower() 
                      or keyword in r.get("input_text", "").lower()]
        
        # 限制数量
        records = records[:limit]
        
        return {"status": "success", "records": records}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v2/history/{record_id}")
async def get_history_detail(record_id: str):
    """获取历史记录详情"""
    try:
        records = load_history()
        for record in records:
            if record["record_id"] == record_id:
                return {"status": "success", **record}
        raise HTTPException(status_code=404, detail="记录不存在")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/generate-from-audio")
async def generate_from_audio(file: UploadFile = File(...), record_type: str = "admission_note"):
    """从音频生成病历（本地Whisper离线识别）"""
    try:
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="空音频文件")
        
        # Whisper转写
        from perception import transcribe_audio
        transcript = transcribe_audio(audio_data)
        transcribed_text = transcript["text"]
        
        if not transcribed_text:
            return {
                "status": "warning",
                "message": "未能识别到有效语音内容，请重试",
                "transcript": "",
            }
        
        # 转写文本送入病历生成管线
        result = agent.process_input(transcribed_text, record_type)
        
        if result["status"] == "awaiting_input":
            return {
                "status": "awaiting_input",
                "transcript": transcribed_text,
                "message": result["message"],
            }
        
        # 保存到历史记录
        add_to_history(
            record_id=result["record"].record_id,
            record_type=record_type,
            content=result["formatted_output"],
            input_text=transcribed_text,
        )
        
        return {
            "status": "success",
            "transcript": transcribed_text,
            "record": {
                "record_id": result["record"].record_id,
                "chief_complaint": result["record"].chief_complaint,
                "present_illness": result["record"].present_illness,
                "physical_exam": result["record"].physical_exam,
                "auxiliary_exam": result["record"].auxiliary_exam,
                "preliminary_diagnosis": result["record"].preliminary_diagnosis,
            },
            "formatted_output": result["formatted_output"],
            "qc_info": result.get("qc_info"),
            "disclaimer": result.get("disclaimer"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


@app.post("/api/v1/speech-to-text")
async def speech_to_text(file: UploadFile = File(...)):
    """仅语音转文字，不生成病历（供前端编辑后手动生成）"""
    try:
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="空音频文件")
        
        from perception import transcribe_audio
        transcript = transcribe_audio(audio_data)
        
        return {
            "status": "success",
            "text": transcript["text"],
            "duration": transcript.get("duration", 0),
            "language": transcript.get("language", "zh"),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


@app.post("/api/v1/chat")
async def chat(input_text: str):
    result = agent.process_input(input_text)
    
    if result["status"] == "awaiting_input":
        return {"role": "assistant", "content": result["message"]}
    
    response_parts = []
    response_parts.append(result["formatted_output"])
    
    if result["disclaimer"]:
        response_parts.append("\n" + result["disclaimer"])
    
    return {"role": "assistant", "content": "\n".join(response_parts)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
