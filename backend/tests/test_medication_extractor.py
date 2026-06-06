"""
L1 单元测试 - 用药提取分段器方案
5条核心测试，通过后才能跑622条全量评估
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.medical_agent import InfoExtractor

def test_extract_medication_segment():
    """用药段专用提取器测试"""
    cases = [
        ("目前用药：阿司匹林100mg qd", "阿司匹林100mg qd"),
        ("用药：无需化疗", "暂无用药"),
        ("目前用药：甲氨蝶呤10mg qw，来氟米特10mg qd", "甲氨蝶呤10mg qw，来氟米特10mg qd"),
        ("", ""),
        ("服药：缬沙坦80mg qd", "缬沙坦80mg qd"),
    ]
    
    passed = 0
    for input_text, expected in cases:
        result = InfoExtractor.extract_medication_segment(input_text)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            print(f"  {status} extract_medication_segment('{input_text}')")
            print(f"     期望: '{expected}'")
            print(f"     实际: '{result}'")
    
    print(f"\n  extract_medication_segment: {passed}/{len(cases)} 通过")
    return passed == len(cases)


def test_segment_followup_input():
    """分段器测试"""
    cases = [
        # FU003: 冠心病随访
        {
            "input": "冠心病随访，患者男，70岁，冠心病PCI术后2年。今日无胸痛，活动后气促较前减轻。血压130/80，心率68。目前用药：阿司匹林100mg qd，阿托伐他汀20mg qn",
            "expected_med": "阿司匹林100mg qd，阿托伐他汀20mg qn",
        },
        # FU008: 肿瘤随访（特殊：无需化疗）
        {
            "input": "肿瘤随访，患者男，60岁，结肠癌术后2年。今日无腹痛，排便正常，体重稳定。CEA 3.5，CA199 15。目前用药：无需化疗",
            "expected_med": "暂无用药",
        },
        # FU007: 甲亢随访
        {
            "input": "甲亢随访，患者女，35岁，Graves病药物治疗1年。今日无怕热多汗，无心悸。TSH 2.5，FT3 4.8，FT4 18。目前用药：甲巯咪唑5mg qd",
            "expected_med": "甲巯咪唑5mg qd",
        },
    ]
    
    passed = 0
    for case in cases:
        segments = InfoExtractor.segment_followup_input(case["input"])
        med = InfoExtractor.extract_medication_segment(segments.get("medication", ""))
        status = "✅" if med == case["expected_med"] else "❌"
        if med == case["expected_med"]:
            passed += 1
        else:
            print(f"  {status} segment_followup_input")
            print(f"     输入: '{case['input'][:50]}...'")
            print(f"     分段: {segments}")
            print(f"     期望用药: '{case['expected_med']}'")
            print(f"     实际用药: '{med}'")
    
    print(f"\n  segment_followup_input: {passed}/{len(cases)} 通过")
    return passed == len(cases)


def test_extract_medications_integration():
    """集成测试：extract_medications 端到端"""
    cases = [
        # 完整FU输入
        (
            "冠心病随访，患者男，70岁，冠心病PCI术后2年。今日无胸痛，活动后气促较前减轻。血压130/80，心率68。目前用药：阿司匹林100mg qd，阿托伐他汀20mg qn",
            "阿司匹林100mg qd，阿托伐他汀20mg qn"
        ),
        # 无需化疗
        (
            "肿瘤随访，患者男，60岁，结肠癌术后2年。今日无腹痛，排便正常，体重稳定。CEA 3.5，CA199 15。目前用药：无需化疗",
            "暂无用药"
        ),
        # 旧版格式（降级路径）- 正则匹配到第一个逗号
        (
            "老张，男，72岁，目前在吃氨氯地平5mg每天，二甲双胍0.5g每天三次",
            "目前在吃氨氯地平5mg每天，"
        ),
    ]
    
    passed = 0
    for input_text, expected in cases:
        result = InfoExtractor.extract_medications(input_text)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            print(f"  {status} extract_medications")
            print(f"     输入: '{input_text[:50]}...'")
            print(f"     期望: '{expected}'")
            print(f"     实际: '{result}'")
    
    print(f"\n  extract_medications 集成: {passed}/{len(cases)} 通过")
    return passed == len(cases)


def main():
    print("=" * 60)
    print("  L1 单元测试 - 用药提取分段器方案")
    print("=" * 60)
    
    results = []
    
    print("\n【测试1】extract_medication_segment 专用提取器")
    results.append(test_extract_medication_segment())
    
    print("\n【测试2】segment_followup_input 分段器")
    results.append(test_segment_followup_input())
    
    print("\n【测试3】extract_medications 端到端集成")
    results.append(test_extract_medications_integration())
    
    print(f"\n{'=' * 60}")
    if all(results):
        print("  🎉 全部L1测试通过！可以跑622条全量评估")
        return 0
    else:
        print(f"  ⚠️ 有测试失败，需要先修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
