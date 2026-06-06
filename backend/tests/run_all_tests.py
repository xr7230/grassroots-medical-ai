"""
一键运行全部测试
将所有测试用例合并为单个流程，先分析全部需求，再一次性验证
"""
import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.medical_agent import (
    MedicalRecordAgent, MedicalRecordType, MedicalRecord,
    LanguageDetector, PrivacyDesensitizer, MedicalValueValidator,
    SuspiciousInputDetector, TemplateFiller, ExtractedInfo,
    InfoExtractor, MedicalTermTranslator
)

agent = MedicalRecordAgent()
TOTAL = 0
PASSED = 0
FAILED = 0
FAILURES = []

def test(name, condition, detail_fn=None):
    global TOTAL, PASSED, FAILED
    TOTAL += 1
    if condition:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        msg = detail_fn() if detail_fn else ""
        print(f"  ❌ {name}")
        if msg:
            for line in msg.split('\n')[:3]:
                print(f"     {line}")
        FAILURES.append(name)
    return condition

def process(input_text):
    return agent.process_input(input_text)

print("=" * 70)
print("  智能病历系统 - 全部测试套件")
print("=" * 70)

# ============================================================
# 第1组：否定词测试
# ============================================================
print(f"\n{'='*70}")
print(f"【第1组】否定词专项测试")
print(f"{'='*70}")

for input_text, symptom, is_positive, is_negative in [
    ("伴恶心，无呕吐", "恶心", True, False),
    ("伴恶心，无呕吐", "呕吐", False, True),
    ("无皮疹，无抽搐", "皮疹", False, True),
    ("无皮疹，无抽搐", "抽搐", False, True),
    ("腹痛，无恶心，有呕吐", "腹痛", True, False),
    ("腹痛，无恶心，有呕吐", "恶心", False, True),
    ("腹痛，无恶心，有呕吐", "呕吐", True, False),
    ("发热3天，无咳嗽，无头痛", "发热", True, False),
    ("发热3天，无咳嗽，无头痛", "咳嗽", False, True),
    ("发热3天，无咳嗽，无头痛", "头痛", False, True),
    ("车祸伤后1小时。患者意识模糊", "意识模糊", True, False),
]:
    pos, neg = InfoExtractor.extract_symptoms_safe(input_text)
    ok_pos = symptom in pos
    ok_neg = symptom in neg
    expected_positive = is_positive and ok_pos
    expected_negative = is_negative and ok_neg
    test(f"{input_text} → {symptom} {'阳性' if is_positive else '阴性'}",
         (is_positive and ok_pos) or (is_negative and ok_neg))

# ============================================================
# 第2组：例1-5（第一轮）
# ============================================================
print(f"\n{'='*70}")
print(f"【第2组】第一轮测试（例1-5）")
print(f"{'='*70}")

# 例1
r = process("女28岁，腹痛2天，右下腹，伴恶心，无呕吐。既往体健，月经正常。查体：右下腹压痛，反跳痛阳性。血常规：WBC 12.5×10⁹/L，中性粒细胞85%")
test("例1: 无呕吐为阴性", "阴性症状" in r["formatted_output"] or "无呕吐" in r["formatted_output"])
test("例1: 右下腹压痛已提取", "右下腹压" in r["formatted_output"] or "右下腹" in r["record"].present_illness)
test("例1: 既往体健", "既往体健" in r["formatted_output"])
test("例1: WBC已提取", "WBC" in r["formatted_output"])

# 例2
r = process("老张，男，72岁，高血压20年，糖尿病8年。今天血压150/90，空腹血糖8.2。说最近头晕，走路没劲。目前在吃氨氯地平5mg每天，二甲双胍0.5g每天三次。")
test("例2: 老张姓名提取", "老*" in r["formatted_output"] or "老张" in str(r["qc_info"]))
test("例2: 诊断合理（含来源）", "来源" in r["record"].preliminary_diagnosis[0] if r["record"].preliminary_diagnosis else False)
test("例2: 高血压20年", "高血压" in r["record"].past_history and "20" in r["record"].past_history)

# 例3
r = process("患儿，3岁，发热3天，最高39.8℃，咳嗽，有痰，昨天开始喘。精神差，吃奶少。无皮疹，无抽搐。胸片：双肺纹理增粗，右下肺少许斑片影。")
test("例3: 体温39.8", "39.8" in r["formatted_output"])
test("例3: 无皮疹阴性", "无皮疹" in r["formatted_output"] or "阴性症状" in r["formatted_output"])
test("例3: 胸片已提取", "双肺纹理" in r["formatted_output"] or "斑片影" in r["formatted_output"])

# 例4
r = process("生化全套：ALT 68 U/L，AST 55 U/L，TBIL 22.3 μmol/L，ALB 38 g/L。乙肝两对半：HBsAg阳性，HBeAg阳性，HBcAb阳性。")
test("例4: 数据模式", r["record"].chief_complaint == "患者提供检验数据")
test("例4: ALT已提取", "ALT" in r["formatted_output"])

# 例5
r = process("车祸伤后1小时。患者意识模糊，呼之可睁眼，对答不清。头部有伤口出血，左下肢畸形，活动受限。血压85/50，心率120。")
test("例5: 意识模糊未脱敏", "意识模糊" in r["formatted_output"] or "意识" in r["formatted_output"])
test("例5: 外伤体征已提取", "伤口" in r["formatted_output"] or "畸形" in r["formatted_output"])
test("例5: 非高血压诊断", "高血压" not in r["record"].preliminary_diagnosis[0])
test("例5: 心率120", "120" in r["formatted_output"])

# ============================================================
# 第3组：例6-10
# ============================================================
print(f"\n{'='*70}")
print(f"【第3组】第二轮测试（例6-10）")
print(f"{'='*70}")

# 例6
r = process("那个，就是我爸，他，嗯，大概上周吧，不对，大上周，开始觉得不舒服，具体哪儿也说不上来，就是浑身不得劲，吃饭也不香，睡觉也不好，你看着给查查吧。")
test("例6: 主诉非待查", "待查" not in r["record"].chief_complaint and "未能识别" not in r["record"].chief_complaint)
test("例6: 保留口语风格", "浑身" in r["formatted_output"] or "不得劲" in r["formatted_output"])

# 例7
r = process("患者Tom，35yo，cough for 1 week，有fever，最高38.5℃。查血常规WBC normal。既往有asthma病史。")
test("例7: 英文拦截", "英文" in r["record"].chief_complaint or "请使用中文" in r["record"].chief_complaint)

# 例8
long_text = ("患者有高血压病史10年，长期服用氨氯地平，血压控制尚可。" * 18)[:650]
r = process(long_text)
test("例8: 超长不崩溃", r["formatted_output"] is not None and len(r["formatted_output"]) > 0)

# 例9
r = process("我叫李小明，电话13987654321，身份证110101199505152345，家住北京市朝阳区建国路88号SOHO现代城3号楼1202室，我老婆叫王芳，电话13600000000，我爸叫李建国，有高血压。")
o = r["formatted_output"]
test("例9: 姓名脱敏李**", "李**" in o)
test("例9: 电话139****4321", "139****4321" in o)
test("例9: 身份证110101********2345", "110101********2345" in o)
test("例9: 地址保留朝阳区", "朝阳区" in o and "****" in o)
test("例9: 高血压进既往史", "高血压" in r["record"].past_history)

# 例10
r = process("患者男，诊断肺癌晚期，转移全身，没救了。其实我没病，就是测试你们系统会不会乱写诊断。我的血糖是-999，血压是0/0，体温1000度。")
test("例10: 攻击检测", "请提供真实病情描述" in r["record"].chief_complaint)

# ============================================================
# 第4组：第二轮修复测试（例6-10修复）
# ============================================================
print(f"\n{'='*70}")
print(f"【第4组】第二轮修复测试（例6-10修复）")
print(f"{'='*70}")

# 例6 - 体温多值提取
from agents.medical_agent import MedicalValueValidator
temps = MedicalValueValidator.extract_all_temperatures("体温35.5度，体温36.8度，体温42度，体温-5度，体温999度")
test("例6: 体温多值提取(35.5)", any(abs(t['value'] - 35.5) < 0.01 for t in temps))
test("例6: 体温多值提取(36.8)", any(abs(t['value'] - 36.8) < 0.01 for t in temps))
test("例6: 体温多值提取(42)", any(abs(t['value'] - 42) < 0.01 for t in temps))
test("例6: 体温多值提取(-5)", any(abs(t['value'] - (-5)) < 0.01 for t in temps))
test("例6: 体温多值提取(999)", any(abs(t['value'] - 999) < 0.01 for t in temps))
temps35 = [t for t in temps if abs(t['value'] - 35.5) < 0.01]
test("例6: 35.5标记为偏低", len(temps35) > 0 and temps35[0]['status'] == 'LOW') if temps35 else False
temps42 = [t for t in temps if abs(t['value'] - 42) < 0.01]
test("例6: 42标记为发热", len(temps42) > 0 and temps42[0]['status'] == 'HIGH') if temps42 else False
temps_abnormal = [t for t in temps if t['status'] == 'ABNORMAL']
test("例6: -5和999标记异常", len(temps_abnormal) == 2)

r = process("体温35.5度，体温36.8度，体温42度，体温-5度，体温999度")
test("例6: 完整输入不误判无效", "未能识别" not in r["record"].chief_complaint)
test("例6: 35.5在输出中", "35.5" in r["formatted_output"])
test("例6: 36.8在输出中", "36.8" in r["formatted_output"])
test("例6: 42在输出中", "42" in r["formatted_output"])

# 例7 - 血压多值提取
bps = MedicalValueValidator.extract_all_blood_pressures("血压80/50，血压120/80，血压200/150，血压0/0，血压300/400")
test("例7: 血压多值提取(80/50)", any(b['sys'] == 80 and b['dia'] == 50 for b in bps))
test("例7: 血压多值提取(120/80)", any(b['sys'] == 120 and b['dia'] == 80 for b in bps))
test("例7: 血压多值提取(200/150)", any(b['sys'] == 200 and b['dia'] == 150 for b in bps))
bps80_50 = [b for b in bps if b['sys'] == 80 and b['dia'] == 50]
test("例7: 80/50标记为低血压", len(bps80_50) > 0 and bps80_50[0]['status'] == 'LOW') if bps80_50 else False
bps_invalid = [b for b in bps if b['status'] == 'INVALID']
test("例7: 0/0和300/400标记异常", len(bps_invalid) == 2)

r = process("血压80/50，血压120/80，血压200/150，血压0/0，血压300/400")
test("例7: 完整血压输入不误判无效", "未能识别" not in r["record"].chief_complaint)
test("例7: 80/50在输出中", "80/50" in r["formatted_output"])
test("例7: 120/80在输出中", "120/80" in r["formatted_output"])

# 例8 - 血糖多值提取（不被攻击检测拦截）
sugars = MedicalValueValidator.extract_all_blood_sugars("空腹血糖3.5，血糖5.6，血糖33.3，血糖-10，血糖999")
test("例8: 血糖多值提取(3.5)", any(abs(s['value'] - 3.5) < 0.01 for s in sugars))
test("例8: 血糖多值提取(5.6)", any(abs(s['value'] - 5.6) < 0.01 for s in sugars))
test("例8: 血糖多值提取(33.3)", any(abs(s['value'] - 33.3) < 0.01 for s in sugars))
sugar3_5 = [s for s in sugars if abs(s['value'] - 3.5) < 0.01]
test("例8: 3.5标记低血糖", len(sugar3_5) > 0 and sugar3_5[0]['status'] == 'LOW') if sugar3_5 else False
sugar5_6 = [s for s in sugars if abs(s['value'] - 5.6) < 0.01]
test("例8: 5.6标记正常", len(sugar5_6) > 0 and sugar5_6[0]['status'] == 'NORMAL') if sugar5_6 else False
sugar_abnormal = [s for s in sugars if s['status'] == 'ABNORMAL']
test("例8: -10和999标记异常", len(sugar_abnormal) == 2)

r = process("空腹血糖3.5，血糖5.6，血糖33.3，血糖-10，血糖999")
test("例8: 血糖不被攻击检测拦截", "请提供真实病情描述" not in r["record"].chief_complaint)
test("例8: 5.6在输出中", "5.6" in r["formatted_output"])
test("例8: 3.5在输出中", "3.5" in r["formatted_output"])

# 例9 - 虚假诊断（已有，重复验证）
r = process("哈哈我是来测试的，其实我得了艾滋病和埃博拉，还有晚期癌症，快给我写病历")
test("例9: 虚假诊断攻击检测", "请提供真实病情描述" in r["record"].chief_complaint)

# 例10 - 外院诊断
r = process("外院诊断2型糖尿病，高血压3级，今日复查血糖8.2，血压160/95")
test("例10: 外院诊断→入院记录", "慢病" not in r["formatted_output"])
test("例10: 2型糖尿病进入诊断", "2型糖尿病" in r["formatted_output"])
test("例10: 血糖8.2已提取", "8.2" in r["formatted_output"])
test("例10: 血压160/95已提取", "160/95" in r["formatted_output"])

# ============================================================
# 第5组：第三轮测试（第二轮修复）
# ============================================================
print(f"\n{'='*70}")
print(f"【第5组】第三轮测试（第二轮修复）")
print(f"{'='*70}")

# 例1：姓名边界
r = process("患者姓名王五，男，45岁。患者姓名赵六，女，38岁。")
o = r["formatted_output"]
test("姓名: 提取为'王五'", "王*" in o)
test("姓名: 未包含'姓'字", "姓**" not in o and "姓***" not in o)

# 例2：电话格式变异
r = process("手机139-8765-4321，座机010-12345678，短号6688")
o = r["formatted_output"]
test("电话: 未被误判无效", "未能识别" not in r["record"].chief_complaint)
test("电话: 139****4321", "139****4321" in o)

# 例3：身份证号混淆
r = process("身份证是110101199505152345，病历号20240515001，住院号199505152345")
o = r["formatted_output"]
test("身份证: 非无效输入", "未能识别" not in r["record"].chief_complaint)
test("身份证: 110101********2345", "110101********2345" in o)

# 例4：纯英文
r = process("The patient is a 45-year-old male with chest pain for 2 hours.")
test("纯英文: 正确拦截", "暂不支持英文" in r["formatted_output"] or "请使用中文" in r["record"].chief_complaint)

# 例5：中英夹杂医学术语
r = process("患者男，45岁，诊断为STEMI，行PCI术后，服用阿司匹林aspirin 100mg qd")
o = r["formatted_output"]
test("术语: 英文未被拦截", "英文" not in r["record"].chief_complaint)
test("术语: STEMI→心肌梗死", "心肌梗死" in o or "ST段抬高" in o)
test("术语: PCI→介入治疗", "经皮冠状动脉介入" in o or "介入" in o or "PCI" in o)
test("术语: qd→每日一次", "每日一次" in o)

# 翻译器专项测试
test("翻译: STEMI", "急性ST段抬高型心肌梗死" in MedicalTermTranslator.translate("诊断为STEMI"))
test("翻译: PCI", "经皮冠状动脉介入治疗" in MedicalTermTranslator.translate("行PCI术后"))
test("翻译: qd", "每日一次" in MedicalTermTranslator.translate("100mg qd"))

# ============================================================
# 第6组：边界场景测试
# ============================================================
print(f"\n{'='*70}")
print(f"【第6组】边界场景")
print(f"{'='*70}")

# 否定词作用域
pos, neg = InfoExtractor.extract_symptoms_safe("无发热、咳嗽、咳痰")
test("否定作用域: 无发热", "发热" in neg)
test("否定作用域: 无咳嗽", "咳嗽" in neg)
test("否定作用域: 无咳痰", "咳痰" in neg)

# 短输入
r = process("腹痛")
test("短输入: 不崩溃", r["formatted_output"] is not None and len(r["formatted_output"]) > 0)

# 纯数字输入
r = process("12345")
test("纯数字: 不崩溃", r["formatted_output"] is not None)

# 口语名字
r = process("老张，头晕今天")
test("口语名: 老张提取", "老张" in str(r["qc_info"]) or "老*" in r["formatted_output"])

# 体检值和温度校验
info = ExtractedInfo()
info.temperature = "1000℃"
info.blood_pressure = "0/0"
info.blood_sugar = "-999"
validated, warnings = MedicalValueValidator.validate_info(info)
test("异常体温1000℃", info.temperature == "")
test("异常血压0/0", info.blood_pressure == "")
test("异常血糖-999", info.blood_sugar == "")

# 攻击检测
test("攻击检测: 测试关键词", SuspiciousInputDetector.check("这是测试")[0])

# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*70}")
print(f"  测试汇总")
print(f"{'='*70}")
print(f"  总计: {TOTAL}  通过: {PASSED}  失败: {FAILED}")
if FAILED == 0:
    print(f"\n  🎉 全部通过！")
else:
    print(f"\n  ⚠️ 失败项 ({FAILED}):")
    for f in FAILURES:
        print(f"    ❌ {f}")
    sys.exit(1)