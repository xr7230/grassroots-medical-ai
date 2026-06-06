from agents.medical_agent import MedicalRecordAgent
agent = MedicalRecordAgent()

print('=' * 80)
print('完整验收测试 - 12组测试用例')
print('=' * 80)

test_cases = [
    ('患者男，56岁，主诉胸痛3小时，伴出汗', '对话模式基础提取'),
    ('空腹血糖7.8，餐后2小时11.2，糖化血红蛋白6.5%', '数据模式基础提取'),
    ('血压160/95，头晕2天，既往有高血压病史5年', '混合模式（症状+数据+既往史）'),
    ('患者女，32岁，咳嗽1周，咳黄痰，发热最高38.5度，胸片示右下肺斑片状阴影', '混合模式（症状+影像）'),
    ('摔伤后左腕疼痛肿胀2小时，活动受限', '极简输入（无检验数据）'),
    ('', '空输入提示'),
    ('asdfghjkl12345', '乱码提示'),
    ('Patient male, 45yo, chest pain for 2 hours', '英文拦截'),
    ('患者张三，电话13812345678，身份证号500101199001011234', '隐私脱敏'),
    ('血糖有点高，血压也正常', '模糊描述处理（不编造数值）'),
    ('患者男，68岁，胸闷1个月。补充：加重3天，伴夜间阵发性呼吸困难。既往有高血压10年，糖尿病5年，查BNP 3500', '长文本/多信息混合'),
    ('当地医院诊断2型糖尿病，血糖控制不佳，空腹9.2', '外院诊断引用'),
]

for i, (input_text, description) in enumerate(test_cases, 1):
    print('\n' + '-'*80)
    print(f'【测试{i}】{description}')
    print(f'输入：{input_text}')
    try:
        result = agent.process_input(input_text)
        print(f'状态：{result["status"]}')
        print('格式化输出：')
        print(result['formatted_output'])
        if result['disclaimer']:
            print(result['disclaimer'])
    except Exception as e:
        print(f'错误：{e}')