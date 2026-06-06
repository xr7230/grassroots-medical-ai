"""测试病库加载"""
import sys
sys.path.insert(0, r'd:\ai医疗\版本二\backend')

from agents.medical_agent import _load_diagnosis_library

groups = _load_diagnosis_library()
print(f'加载了 {len(groups)} 个疾病')
print('前10个疾病:')
for g in groups[:10]:
    print(f"  - {g['diagnosis']} (优先级: {g['priority']}, 关键词: {len(g['keywords'])}个)")
