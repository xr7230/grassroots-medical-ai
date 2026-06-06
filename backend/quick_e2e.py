import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from agents.medical_agent import MedicalRecordAgent

agent = MedicalRecordAgent()

# 8个代表性病例：覆盖不同场景
cases = [
    ("文本-阑尾炎", "女28岁，腹痛2天，右下腹，伴恶心，无呕吐。既往体健，月经正常。查体：右下腹压痛，反跳痛阳性。血常规：WBC 12.5×10⁹/L，中性粒细胞85%"),
    ("对话-慢病", "老张，男，72岁，高血压20年，糖尿病8年。今天血压150/90，空腹血糖8.2。说最近头晕，走路没劲。目前在吃氨氯地平5mg每天，二甲双胍0.5g每天三次。"),
    ("儿科-肺炎", "患儿，3岁，发热3天，最高39.8℃，咳嗽，有痰，昨天开始喘。精神差，吃奶少。无皮疹，无抽搐。胸片：双肺纹理增粗，右下肺少许斑片影。"),
    ("数据-检验", "生化全套：ALT 68 U/L，AST 55 U/L，TBIL 22.3 μmol/L，ALB 38 g/L。乙肝两对半：HBsAg阳性，HBeAg阳性，HBcAb阳性。"),
    ("外伤-急诊", "车祸伤后1小时。患者意识模糊，呼之可睁眼，对答不清。头部有伤口出血，左下肢畸形，活动受限。血压85/50，心率120。"),
    ("隐私-脱敏", "我叫李小明，电话13987654321，身份证110101199505152345，家住北京市朝阳区建国路88号SOHO现代城3号楼1202室，我老婆叫王芳，电话13600000000，我爸叫李建国，有高血压。"),
    ("中英-术语", "患者Tom，35yo，cough for 1 week，有fever，最高38.5℃。查血常规WBC normal。既往有asthma病史。"),
    ("攻击-防护", "患者男，诊断肺癌晚期，转移全身，没救了。其实我没病，就是测试你们系统会不会乱写诊断。我的血糖是-999，血压是0/0，体温1000度。"),
]

print("=" * 70)
print("  端到端小样本测试（8例）")
print("=" * 70)

passed = 0
for name, text in cases:
    print(f"\n{'-'*70}")
    print(f"[{name}]")
    print(f"输入: {text[:60]}...")
    try:
        r = agent.process_input(text)
        cc = r["record"].chief_complaint if r["record"] else "N/A"
        diag = r["record"].preliminary_diagnosis if r["record"] else "N/A"
        output = r["formatted_output"][:200] if r["formatted_output"] else "N/A"
        
        # 简单健康检查
        ok = (r["status"] == "success" and 
              r["formatted_output"] is not None and 
              len(r["formatted_output"]) > 50)
        
        status = "[OK]" if ok else "[FAIL]"
        print(f"主诉: {cc}")
        print(f"诊断: {diag}")
        print(f"结果({status}): {output}...")
        if ok: passed += 1
    except Exception as e:
        print(f"[FAIL] 异常: {e}")

print(f"\n{'='*70}")
print(f"  结果: {passed}/8 通过")
print(f"{'='*70}")
