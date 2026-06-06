"""Quick CLS007/009 test"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from agents.medical_agent import MedicalRecordAgent

agent = MedicalRecordAgent()

for cid, text in [
    ('CLS007', "患者男，68岁，胸闷1个月，加重3天。顺便说下，他血压平时150/90，有高血压10年了"),
    ('CLS009', "BNP 3500，心电图房颤。患者男68岁，胸闷1月加重3天，高血压10年糖尿病5年"),
    ('CLS008', "BNP 三千五，心电图房颤。患者女65岁"),
    ('CLS020', "肿瘤标志物：AFP 450，CEA 12.5，CA199 85。B超：肝右叶占位5×4cm"),
]:
    result = agent.process_input(text)
    diag = result['record'].preliminary_diagnosis
    inp_type = result['qc_info'].get('input_type', '?')
    print(f"[{cid}] type={inp_type} | diag={diag}")