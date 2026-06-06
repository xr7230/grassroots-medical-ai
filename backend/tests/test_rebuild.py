"""
测试脚本 - 验证重构版智能体
"""
import sys
sys.path.append('.')

from agents.medical_agent import MedicalRecordAgent

def test_case_1():
    """测试用例1：胸痛场景"""
    print("=" * 60)
    print("测试用例1：胸痛场景")
    print("=" * 60)
    print("输入：患者男，56岁，主诉胸痛3小时，伴出汗")
    print()
    
    agent = MedicalRecordAgent()
    result = agent.process_input("患者男，56岁，主诉胸痛3小时，伴出汗")
    
    print("输出：")
    print(result["formatted_output"])
    print()
    
    # 检查提取结果
    qc_info = result.get("qc_info", {})
    extracted_info = qc_info.get("extracted_info", {})
    print("提取信息：")
    print(f"  性别: {extracted_info.get('gender', '未提取')}")
    print(f"  年龄: {extracted_info.get('age', '未提取')}")
    print(f"  主诉: {extracted_info.get('chief_complaint', '未提取')}")
    print(f"  症状: {extracted_info.get('symptoms', '未提取')}")
    print()
    
    # 检查质控问题
    issues = qc_info.get("issues", [])
    if issues:
        print("质控问题：")
        for issue in issues:
            print(f"  - {issue['message']}: {issue['suggestion']}")
    else:
        print("质控问题：无")


def test_case_2():
    """测试用例2：糖尿病随访"""
    print("\n" + "=" * 60)
    print("测试用例2：糖尿病血糖数据提取")
    print("=" * 60)
    print("输入：空腹血糖7.8，餐后2小时11.2，糖化血红蛋白6.5%")
    print()
    
    agent = MedicalRecordAgent()
    result = agent.process_input("空腹血糖7.8，餐后2小时11.2，糖化血红蛋白6.5%")
    
    print("输出：")
    print(result["formatted_output"])
    print()
    
    # 检查提取结果
    qc_info = result.get("qc_info", {})
    extracted_info = qc_info.get("extracted_info", {})
    print("提取信息：")
    print(f"  空腹血糖: {extracted_info.get('blood_sugar', '未提取')}")
    print(f"  餐后血糖: {extracted_info.get('blood_sugar_2h', '未提取')}")
    print(f"  糖化血红蛋白: {extracted_info.get('hba1c', '未提取')}")
    print(f"  疾病类型: {extracted_info.get('disease_type', '未提取')}")
    print()
    
    # 检查质控问题
    issues = qc_info.get("issues", [])
    if issues:
        print("质控问题：")
        for issue in issues:
            print(f"  - {issue['message']}: {issue['suggestion']}")
    else:
        print("质控问题：无")


def test_case_3():
    """测试用例3：完整信息输入"""
    print("\n" + "=" * 60)
    print("测试用例3：完整信息输入")
    print("=" * 60)
    print("输入：患者女，48岁，头痛2天，伴恶心呕吐，血压145/95 mmHg")
    print()
    
    agent = MedicalRecordAgent()
    result = agent.process_input("患者女，48岁，头痛2天，伴恶心呕吐，血压145/95 mmHg")
    
    print("输出：")
    print(result["formatted_output"])
    print()
    
    # 检查提取结果
    qc_info = result.get("qc_info", {})
    extracted_info = qc_info.get("extracted_info", {})
    print("提取信息：")
    print(f"  性别: {extracted_info.get('gender', '未提取')}")
    print(f"  年龄: {extracted_info.get('age', '未提取')}")
    print(f"  主诉: {extracted_info.get('chief_complaint', '未提取')}")
    print(f"  血压: {extracted_info.get('blood_pressure', '未提取')}")
    print(f"  症状: {extracted_info.get('symptoms', '未提取')}")


if __name__ == "__main__":
    test_case_1()
    test_case_2()
    test_case_3()
    print("\n" + "=" * 60)
    print("测试完成！")