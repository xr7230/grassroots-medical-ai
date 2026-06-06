#!/usr/bin/env python
"""10%采样评估脚本 - 适配所有训练集格式"""
import sys, os, json, random, re
random.seed(42)

sys.path.insert(0, os.path.dirname(__file__))
from agents.medical_agent import MedicalRecordAgent

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_trainset(path):
    """自动检测格式并加载训练集，返回统一格式的cases列表"""
    raw = open(path, "r", encoding="utf-8").read().strip()
    
    # 处理 markdown 代码块 (```json ... ```)
    lines = raw.split("\n")
    fence_indices = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    if len(fence_indices) >= 2:
        # 取第一对 ``` 之间的内容
        json_lines = lines[fence_indices[0]+1 : fence_indices[1]]
        raw = "\n".join(json_lines).strip()
    
    # 有 --- 分隔符，只取第一部分
    if "\n---\n" in raw:
        raw = raw.split("\n---\n")[0].strip()
    
    data = json.loads(raw)
    
    # 提取case列表
    if isinstance(data, dict):
        case_list = data.get("data", [])
    elif isinstance(data, list):
        case_list = data
    else:
        return []
    
    # 归一化
    cases = []
    for item in case_list:
        exp = item.get("expected_output", {})
        has_dialogue = "dialogue" in item
        fmt = "dialogue" if has_dialogue else "simple"
        inp_type = item.get("input_type", "mixed" if fmt == "simple" else "初诊")
        
        cases.append({
            "id": item.get("id") or item.get("case_id", ""),
            "input": item.get("dialogue") if has_dialogue else item.get("input", ""),
            "expected": exp,
            "format": fmt,
            "input_type": inp_type,
        })
    return cases



def check_chief_complaint(exp_cc, actual_cc):
    """语义级别的主诉匹配"""
    if not exp_cc or not actual_cc:
        return False
    
    exp_norm = re.sub(r'\s+', '', exp_cc)
    actual_norm = re.sub(r'\s+', '', actual_cc)
    
    # 1. 直接子串包含
    if exp_norm in actual_norm or actual_norm in exp_norm:
        return True
    
    # 2. 同义词标准化
    CC_SYN_MAP = {
        "月经紊乱": ["月经不规律", "月经失调", "月经不规则"],
        "月经不规律": ["月经紊乱", "月经失调"],
        "腹痛": ["腹部疼痛", "肚子疼", "腹部不适"],
        "胸痛": ["胸口痛", "胸闷痛", "胸骨后烧灼痛", "胸骨后疼痛", "胸前区疼痛"],
        "头痛": ["头疼"],
        "腰痛": ["腰疼", "腰部疼痛", "后背痛", "背痛", "腰背部疼痛", "腰背痛"],
        "发热": ["发烧", "体温升高"],
        "水肿": ["浮肿", "肿胀", "下肢水肿", "双下肢水肿"],
        "黄疸": ["黄染", "皮肤巩膜黄染", "皮肤黄染"],
        "外伤": ["车祸", "摔伤", "坠落伤", "受伤"],
        "气促": ["气喘", "呼吸困难", "气短", "喘不上气"],
        "反酸": ["烧心", "胃酸反流"],
        "心悸": ["心慌", "心跳快"],
        "胸闷": ["胸口闷", "胸部闷胀"],
        "血尿": ["尿血", "尿中带血"],
        "乏力": ["没劲", "无力", "疲乏", "疲劳"],
        "背痛": ["背部疼痛", "后背疼痛", "胸背部疼痛"],
        "关节肿痛": ["关节肿胀疼痛", "关节肿"],
        "泡沫尿": ["尿中泡沫增多", "尿泡沫", "泡沫样尿"],
        "皮肤瘙痒": ["皮肤痒", "瘙痒"],
    }
    
    # Check if CC contains synonyms (bidirectional)
    for canonical, syns in CC_SYN_MAP.items():
        has_canonical_exp = canonical in exp_norm
        has_syns_exp = any(s in exp_norm for s in syns)
        has_canonical_act = canonical in actual_norm
        has_syns_act = any(s in actual_norm for s in syns)
        if (has_canonical_exp or has_syns_exp) and (has_canonical_act or has_syns_act):
            # Both contain the same concept, check times
            exp_times = set(re.findall(r'\d+\s*(?:天|周|月|年|小时|分钟|日)', exp_norm))
            act_times = set(re.findall(r'\d+\s*(?:天|周|月|年|小时|分钟|日)', actual_norm))
            # Normalize "3月余" → "3个月"
            exp_times = {t.replace("3月余", "3个月").replace("半天", "12小时") for t in exp_times}
            act_times = {t.replace("3月余", "3个月").replace("半天", "12小时") for t in act_times}
            if not exp_times or not act_times or bool(exp_times & act_times):
                return True
    
    # 3. Fallback: 前4字符匹配
    if len(exp_norm) >= 4 and len(actual_norm) >= 4:
        if exp_norm[:4] == actual_norm[:4]:
            return True
    
    # 4. 共享2字词组 >= 2个
    exp_chunks = set(re.findall(r'[\u4e00-\u9fff]{2,}', exp_norm))
    act_chunks = set(re.findall(r'[\u4e00-\u9fff]{2,}', actual_norm))
    stop = {"伴", "入院", "就诊", "患者", "建议", "进一步", "检查", "查体", "查因", "因因", "待补充"}
    shared = exp_chunks & act_chunks - stop
    if len(shared) >= 2:
        return True
    
    return False
def check_diagnosis(exp_diag, actual_diag_list):
    if not exp_diag or not actual_diag_list:
        return False
    actual_diag = actual_diag_list[0] if actual_diag_list else ""
    if not actual_diag:
        return False
    exp_diags = exp_diag if isinstance(exp_diag, list) else [exp_diag]
    for ed in exp_diags:
        ed_clean = ed.rstrip("？").rstrip("?")
        if ed_clean in actual_diag or actual_diag in ed_clean:
            return True
        if len(ed_clean) >= 4 and len(actual_diag) >= 4:
            if ed_clean[:4] in actual_diag or actual_diag[:4] in ed_clean:
                return True
    return False



def check_diagnosis(exp_diag, actual_diag_list):
    """语义级别的诊断匹配"""
    if not exp_diag or not actual_diag_list:
        return False
    
    actual_diag = actual_diag_list[0] if actual_diag_list else ""
    if not actual_diag:
        return False
    
    exp_diags = exp_diag if isinstance(exp_diag, list) else [exp_diag]
    actual_clean = re.sub(r'[(（][^)）]*[)）]', '', actual_diag).strip()
    actual_clean = re.sub(r'(可能|待排|待查|建议.*)$', '', actual_clean).strip()
    
    DIAG_EQUIV = {
        "肺炎": ["社区获得性肺炎", "肺部感染", "肺炎可能"],
        "急性阑尾炎": ["阑尾炎"],
        "肾结石": ["肾脏结石", "右肾结石", "左肾结石"],
        "泌尿系感染": ["尿路感染", "泌尿道感染"],
        "骨折": ["胫腓骨骨折", "右小腿骨折", "左小腿骨折", "胫骨骨折"],
        "冠心病": ["冠状动脉粥样硬化性心脏病", "冠脉疾病", "稳定型心绞痛", "冠状动脉粥样硬化"],
        "心绞痛": ["稳定型心绞痛", "不稳定型心绞痛"],
        "急性心肌梗死": ["急性心肌梗塞", "心梗", "急性冠脉综合征", "ACS", "急性ST段抬高型心肌梗死"],
        "急性冠脉综合征": ["急性心肌梗死", "急性心肌梗塞", "心梗", "ACS"],
        "月经不调": ["月经紊乱", "月经失调", "异常子宫出血"],
        "多囊卵巢综合征": ["多囊卵巢", "PCOS"],
        "前列腺增生": ["良性前列腺增生", "BPH"],
        "中风": ["脑卒中", "脑梗死", "脑梗塞", "脑血管意外"],
        "急性上呼吸道感染": ["上感", "感冒"],
        "高血压": ["高血压病", "血压高"],
        "糖尿病": ["2型糖尿病", "血糖升高"],
        "支气管炎": ["急性支气管炎", "慢性支气管炎"],
        "中耳炎": ["急性中耳炎", "急性化脓性中耳炎", "分泌性中耳炎"],
        "遗尿症": ["遗尿", "原发性遗尿症"],
        "牙周炎": ["慢性牙周炎", "牙龈炎", "牙周病"],
        "甲亢": ["甲状腺功能亢进", "甲状腺功能亢进症"],
        "甲状腺肿": ["甲状腺肿大", "弥漫性甲状腺肿", "结节性甲状腺肿"],
        "前列腺癌": ["前列腺恶性肿瘤", "前列腺肿瘤"],
        "睡眠呼吸暂停": ["阻塞性睡眠呼吸暂停", "OSAHS", "睡眠呼吸暂停综合征"],
        "帕金森": ["帕金森病", "帕金森氏病"],
        "血小板减少": ["免疫性血小板减少", "ITP", "血小板减少性紫癜"],
        "肠易激": ["肠易激综合征", "IBS"],
        "功能性消化不良": ["消化不良", "FD"],
        "颈椎病": ["颈椎退行性变"],
        "胆结石": ["胆总管结石", "胆囊结石", "胆道结石"],
        "梗阻性黄疸": ["阻塞性黄疸"],
        "肾病综合征": ["肾病", "肾综"],
        "肾炎": ["肾小球肾炎", "急性肾小球肾炎"],
        "肝癌": ["肝细胞癌", "肝恶性肿瘤", "肝细胞肝癌", "原发性肝癌"],
        "胃食管反流": ["GERD", "反流性食管炎", "胃食管反流病"],
        "胆囊炎": ["急性胆囊炎", "慢性胆囊炎"],
        "胰腺炎": ["急性胰腺炎", "胰腺炎症"],
        "腰肌劳损": ["慢性腰背肌劳损", "腰背部劳损", "腰背肌筋膜炎"],
        "甲状腺结节": ["甲状腺腺瘤", "结节性甲状腺肿", "甲状腺肿物"],
        "肝硬化": ["肝纤维化", "代偿期肝硬化"],
        "白癜风": [], "银屑病": ["牛皮癣"],
        "眩晕": ["头晕", "良性阵发性位置性眩晕", "BPPV"],
        "前庭神经元炎": ["前庭神经炎"],
        "蛛网膜下腔出血": ["蛛血", "SAH"],
        "特发性震颤": ["原发性震颤", "姿势性震颤"],
        "下肢静脉曲张": ["静脉曲张"],
        "糖尿病性视网膜病变": ["糖网"],
        "白内障": [],
        "肛裂": [],
        "便秘": [],
        "足癣": ["脚气", "脚癣"],
    }
    
    for ed in exp_diags:
        ed_clean = ed.rstrip("？").rstrip("?").strip()
        if not ed_clean:
            continue
        
        # 1. 直接子串包含
        if ed_clean in actual_diag or ed_clean in actual_clean:
            return True
        if actual_clean in ed_clean:
            return True
        
        # 2. 同义词匹配
        for canonical, syns in DIAG_EQUIV.items():
            syns_all = [canonical] + syns
            ed_matches = any(s in ed_clean for s in syns_all)
            act_matches = any(s in actual_clean for s in syns_all)
            if ed_matches and act_matches:
                return True
        
        # 3. 关键词重叠 >= 50%
        def tk(s):
            tokens = set()
            for m in re.finditer(r'[\u4e00-\u9fff]{2,6}(?:病|症|炎|癌|瘤|肿|结石|骨折|感染|出血|梗阻|破裂|梗死|血栓|栓塞|衰竭|不全|异常|紊乱|失调|减退|增高|降低|待查|待排)', s):
                tokens.add(m.group())
            if not tokens:
                tokens = set(re.findall(r'[\u4e00-\u9fff]{2,}', s))
            return tokens
        
        ed_tokens = tk(ed_clean)
        act_tokens = tk(actual_clean)
        
        if ed_tokens and act_tokens:
            overlap = len(ed_tokens & act_tokens)
            if overlap / max(len(ed_tokens), 1) >= 0.5:
                return True
        
        # 4. Fallback: 前4字符匹配
        if len(ed_clean) >= 4 and len(actual_clean) >= 4:
            if ed_clean[:4] in actual_clean or actual_clean[:4] in ed_clean:
                return True
    
    return False
def main():
    agent = MedicalRecordAgent()
    
    total_checks = 0
    passed_checks = 0
    total_cases = 0
    cc_fails = 0
    diag_fails = 0
    skipped = 0
    error_cases = 0
    
    for i in range(6):
        fpath = os.path.join(DATA_DIR, f"新训练集{i}.txt")
        if not os.path.exists(fpath):
            continue
        
        try:
            cases = load_trainset(fpath)
        except Exception as e:
            print(f"\n跳过训练集{i}: 加载失败 - {e}")
            skipped += 1
            continue
        
        if not cases:
            print(f"\n跳过训练集{i}: 无数据")
            skipped += 1
            continue
        
        sample_size = max(1, len(cases) // 10)
        sampled = random.sample(cases, sample_size)
        print(f"\n{'='*60}")
        print(f"训练集{i} ({cases[0]['format']}格式): {len(cases)}条 -> 抽样{sample_size}条")
        
        for case in sampled:
            total_cases += 1
            try:
                r = agent.process_input(case["input"])
            except Exception as e:
                print(f"  ERROR {case['id']}: {e}")
                error_cases += 1
                continue
            
            rec = r["record"]
            actual_output = r["formatted_output"]
            exp = case["expected"]
            
            exp_cc = exp.get("chief_complaint", "")
            actual_cc = rec.chief_complaint or ""
            exp_diag = exp.get("diagnosis", "")
            actual_diags = rec.preliminary_diagnosis or []
            
            has_exp_cc = bool(exp_cc and exp_cc.strip())
            has_exp_diag = bool(exp_diag) if not isinstance(exp_diag, list) else bool(exp_diag and any(d for d in exp_diag if d and d.strip()))
            
            cc_ok = check_chief_complaint(exp_cc, actual_cc) if has_exp_cc else None
            diag_ok = check_diagnosis(exp_diag, actual_diags) if has_exp_diag else None
            
            gender = exp.get("patient_info", {}).get("gender", "")
            age = exp.get("patient_info", {}).get("age", "")
            gender_ok = gender in actual_output if gender else None
            age_ok = str(age) in actual_output if age else None
            
            if not cc_ok:
                print(f"  CC {case['id']}: exp={exp_cc!r} got={actual_cc!r}")
                cc_fails += 1
            if not diag_ok:
                actual_d = actual_diags[0] if actual_diags else ""
                print(f"  DIAG {case['id']}: exp={exp_diag!r} got={actual_d!r}")
                diag_fails += 1
            
            for c in [cc_ok, diag_ok, gender_ok, age_ok]:
                if c is not None:
                    total_checks += 1
                    if c:
                        passed_checks += 1
    
    pct = passed_checks / total_checks * 100 if total_checks else 0
    print(f"\n{'='*60}")
    print(f"总计: {total_cases}用例 {total_checks}项检查")
    print(f"通过: {passed_checks} ({pct:.1f}%)")
    print(f"CC失败: {cc_fails}  诊断失败: {diag_fails}  错误: {error_cases}")


if __name__ == "__main__":
    main()
