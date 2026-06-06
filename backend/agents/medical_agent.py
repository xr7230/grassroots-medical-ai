"""
AI医疗文书智能体 - 版本三（重构版）
采用分而治之策略：先提取、再校验、最后填模板
核心改进：消除幻觉、提升信息提取准确率
"""
import re
import json
import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
from loguru import logger
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# === API Key dynamic injection ===
_api_key_override = None

def set_api_key(key):
    global _api_key_override
    _api_key_override = key if key else None

def get_api_key():
    return _api_key_override or os.getenv("DEEPSEEK_API_KEY", "")


def _get_disease_category(disease_name: str) -> str:
    """获取疾病所属类别（用于输入类型感知匹配）"""
    disease_name = disease_name.lower()
    
    # 呼吸系统
    if any(k in disease_name for k in ['肺炎', '支气管炎', '哮喘', '慢阻肺', '肺栓塞', '气胸', '肺癌', '肺结核', '呼吸', '咳嗽', '咳痰', '喘息', '气急', '鼻塞', '流涕', '喷嚏', '咽痛', '扁桃体', '鼻窦']):
        return 'respiratory'
    
    # 心血管系统
    if any(k in disease_name for k in ['心梗', '冠脉', '心绞痛', '心衰', '高血压', '心律失常', '房颤', '房扑', '早搏', '心肌', '心包', '心脏', '主动脉', '静脉血栓', '动脉栓塞', '冠心病']):
        return 'cardiovascular'
    
    # 神经系统
    if any(k in disease_name for k in ['脑梗', '脑出血', '蛛网膜', '脑膜', '脑炎', '癫痫', '帕金森', '阿尔茨海默', '痴呆', '头痛', '头晕', '眩晕', '神经', '脊髓', '格林巴利', '重症肌无力', '多发性硬化']):
        return 'neurological'
    
    # 消化系统
    if any(k in disease_name for k in ['阑尾炎', '胰腺炎', '胆囊炎', '胆石', '肝炎', '肝硬化', '肝癌', '溃疡', '胃炎', '肠炎', '胃肠', '肠梗阻', '肠结核', '克罗恩', '溃疡性结肠', '痔疮', '肛裂', '肛瘘', '食管癌', '胃癌', '肠癌', '胰腺癌', '肝损伤', '脂肪肝', '食管']):
        return 'gastrointestinal'
    
    # 内分泌代谢
    if any(k in disease_name for k in ['糖尿病', '甲亢', '甲减', '甲状腺', '库欣', '醛固酮', '骨质疏松', '痛风', '高尿酸', '低血糖', '高脂血症', '肥胖', '代谢综合征', '酮症酸中毒', '高渗', '甲状旁腺', '肾上腺', '肢端肥大']):
        return 'endocrine'
    
    # 泌尿系统
    if any(k in disease_name for k in ['肾炎', '肾病', '肾衰竭', '尿毒症', '肾结石', '肾囊肿', '肾癌', '膀胱炎', '肾盂肾炎', '前列腺', '膀胱癌', '睾丸', '精索', '附睾炎', '鞘膜积液', '尿路感染']):
        return 'urinary'
    
    # 血液系统
    if any(k in disease_name for k in ['贫血', '白血病', '淋巴瘤', '骨髓瘤', '骨髓增生', '血小板减少', '血友病', '紫癜', '红细胞增多', '血小板增多', '溶血', '凝血功能障碍', '弥散性血管内凝血']):
        return 'hematological'
    
    # 风湿免疫
    if any(k in disease_name for k in ['类风湿', '红斑狼疮', '强直', '骨关节炎', '干燥综合征', '硬化症', '肌炎', '皮肌炎', '血管炎', '白塞病', '结缔组织病', '抗磷脂综合征', '多软骨炎', '斯蒂尔病', 'IgG4']):
        return 'rheumatological'
    
    # 感染性疾病
    if any(k in disease_name for k in ['流感', '感冒', '新冠', '麻疹', '水痘', '腮腺炎', '风疹', '手足口', '猩红热', '百日咳', '急疹', '轮状病毒', '痢疾', '伤寒', '疟疾', '狂犬病', '破伤风', '艾滋病', '梅毒', '淋病', '疱疹', '湿疣', '软下疳', '登革热', '乙脑', '流脑', '钩体病', '恙虫病', '布鲁菌', '出血热', '单核细胞增多', '川崎病']):
        return 'infectious'
    
    # 肿瘤
    if any(k in disease_name for k in ['癌', '肿瘤', '肉瘤', '淋巴瘤', '白血病', '骨髓瘤']):
        return 'oncological'
    
    # 妇产科
    if any(k in disease_name for k in ['早孕', '妊娠', '异位妊娠', '宫外孕', '流产', '早产', '前置胎盘', '胎盘早剥', '子痫', '胎儿', '胎动', '胎膜早破', '过期妊娠', '产后', '产褥', '乳腺', '子宫', '卵巢', '宫颈', '内膜', '多囊卵巢', '盆腔炎', '阴道炎', '宫颈炎', '痛经']):
        return 'obstetrics_gynecology'
    
    # 儿科
    if any(k in disease_name for k in ['新生儿', '脑瘫', '婴儿痉挛', '小儿腹泻', '佝偻病', '高热惊厥', '生长痛', '性早熟', '遗尿', '多动', '抽动', '分离焦虑']):
        return 'pediatric'
    
    # 骨科运动
    if any(k in disease_name for k in ['骨折', '腰椎', '颈椎', '肩周', '网球肘', '腱鞘', '滑囊', '半月板', '韧带', '踝', '髌骨', '足底筋膜', '腰肌', '骨质疏松', '骨软骨瘤', '骨肉瘤', '骨髓炎', '胸廓出口', '腕管', '肘管', '关节脱位', '脊柱', '脊髓', '软组织', '挤压综合征', '脂肪栓塞']):
        return 'orthopedic'
    
    # 眼科
    if any(k in disease_name for k in ['白内障', '青光眼', '结膜炎', '角膜炎', '葡萄膜炎', '视神经', '视网膜', '黄斑', '干眼', '屈光', '近视', '远视', '散光', '翼状胬肉', '老视', '眼睑', '倒睫']):
        return 'ophthalmological'
    
    # 耳鼻喉
    if any(k in disease_name for k in ['中耳炎', '梅尼埃', '耳聋', '眩晕', '鼻炎', '鼻窦炎', '鼻出血', '咽炎', '扁桃体', '腺样体', '喉炎', '会厌炎', '声带', '鼻咽癌', '鼻腔异物', '耵聍', '外耳道', '耳廓', '嗅觉']):
        return 'ent'
    
    # 口腔科
    if any(k in disease_name for k in ['龋病', '牙髓', '根尖', '牙周', '牙龈', '智齿', '口腔溃疡', '白斑', '扁平苔藓', '念珠菌', '颌面部', '涎石', '颌骨骨髓炎', '颞下颌']):
        return 'dental'
    
    # 皮肤科
    if any(k in disease_name for k in ['湿疹', '皮炎', '银屑病', '荨麻疹', '药疹', '带状疱疹', '疱疹', '疣', '痤疮', '脂溢性', '黄褐斑', '白癜风', '斑秃', '脱发', '甲沟炎', '脓疱疮', '疖', '痈', '蜂窝织炎', '丹毒', '癣', '疥疮', '虱病', '梅毒疹', '基底细胞癌', '鳞状细胞癌', '黑色素瘤', '玫瑰糠疹', '扁平苔藓', '毛发苔藓', '多汗', '腋臭', '瘢痕', '脂肪瘤', '皮脂腺']):
        return 'dermatological'
    
    # 精神心理
    if any(k in disease_name for k in ['抑郁', '焦虑', '强迫', '双相', '精神分裂', '失眠', '睡眠', '创伤后应激', '躯体形式', '转换障碍', '适应障碍', '进食障碍', '厌食', '贪食', '酒精依赖', '物质依赖', '成瘾', '认知障碍', '痴呆', '谵妄']):
        return 'psychiatric'
    
    # 中毒急救
    if any(k in disease_name for k in ['中毒', '中暑', '溺水', '电击', '烧伤', '冻伤']):
        return 'toxicology'
    
    # 创伤外科
    if any(k in disease_name for k in ['外伤', '多发伤', '复合伤', '颅脑损伤', '脑震荡', '脑挫裂伤', '血肿', '头皮', '气胸', '血胸', '肺挫伤', '肋骨骨折', '血气胸', '肝破裂', '脾破裂', '肾挫伤', '肠穿孔', '腹膜后血肿', '骨盆骨折', '创伤性休克']):
        return 'trauma'
    
    # 其他
    return 'other'



class MedicalRecordType(Enum):
    ADMISSION_NOTE = "admission_note"
    OUTPATIENT_NOTE = "outpatient_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    FOLLOW_UP_RECORD = "follow_up_record"


@dataclass
class ExtractedInfo:
    """提取的结构化信息"""
    # 基本信息
    gender: str = ""
    age: str = ""
    chief_complaint: str = ""
    
    # 病史信息
    present_illness: str = ""
    past_history: str = ""
    personal_history: str = ""
    family_history: str = ""
    marital_history: str = ""  # 婚育史
    
    # 检查指标
    blood_sugar: str = ""
    blood_sugar_2h: str = ""
    hba1c: str = ""
    blood_pressure: str = ""
    heart_rate: str = ""  # 心率
    respiratory_rate: str = ""  # 呼吸
    temperature: str = ""  # 体温
    
    # 症状
    symptoms: List[str] = field(default_factory=list)
    positive_symptoms: List[str] = field(default_factory=list)  # 阳性症状（无否定词修饰）
    negative_symptoms: List[str] = field(default_factory=list)  # 阴性症状（无XX）
    
    # 疾病类型
    disease_type: str = ""
    
    # 新增字段
    medical_history: List[str] = field(default_factory=list)  # 既往史
    imaging_exam: List[str] = field(default_factory=list)     # 影像检查
    fever_temp: str = ""  # 发热温度
    
    # 诊断相关
    has_diagnosis_source: bool = False  # 是否有诊断来源
    diagnosis_source: str = ""  # 诊断来源
    
    # 治疗计划
    treatment_plan: str = ""
    
    # 患者基本信息（脱敏用）
    patient_name: str = ""  # 患者姓名
    phone: str = ""  # 联系电话
    id_card: str = ""  # 身份证号
    
    # 新增结构化字段
    physical_exam_raw: str = ""  # 查体原文
    lab_exam_raw: str = ""      # 检验原文
    medications: str = ""       # 用药信息
    pain_location: str = ""     # 疼痛部位/定位
    address: str = ""           # 地址
    family_members: List[str] = field(default_factory=list)  # 家属信息（脱敏前）
    
    # 新增检验指标字段
    blood_oxygen: str = ""      # 血氧饱和度
    tsh: str = ""               # TSH
    ft3: str = ""               # FT3
    ft4: str = ""               # FT4
    cea: str = ""               # CEA
    ca199: str = ""             # CA199
    ca153: str = ""             # CA153乳腺癌肿瘤标志物
    afp: str = ""               # AFP
    hbv_dna: str = ""           # HBV-DNA乙肝病毒DNA
    creatinine: str = ""        # 肌酐
    bun: str = ""               # 尿素氮
    pt: str = ""                # PT凝血酶原时间
    aptt: str = ""              # APTT活化部分凝血活酶时间
    fib: str = ""               # FIB纤维蛋白原
    d_dimer: str = ""           # D-二聚体
    esr: str = ""               # 血沉
    crp: str = ""               # CRP
    alt: str = ""               # ALT谷丙转氨酶
    ast: str = ""               # AST谷草转氨酶
    tc: str = ""                # 总胆固醇
    tg: str = ""                # 甘油三酯
    ldl_c: str = ""             # LDL-C
    hdl_c: str = ""             # HDL-C
    uric_acid: str = ""              # 尿酸
    wbc: str = ""                    # WBC白细胞
    neutrophil_pct: str = ""         # 中性粒细胞百分比
    amylase: str = ""                # 淀粉酶
    bnp: str = ""                    # BNP值
    hb: str = ""                     # 血红蛋白
    
    def is_empty(self) -> bool:
        """检查是否为空"""
        return not any([
            self.gender, self.age, self.chief_complaint,
            self.present_illness, self.past_history,
            self.blood_sugar, self.blood_pressure,
            self.symptoms])



@dataclass
class MedicalRecord:
    record_type: MedicalRecordType
    patient_info: str = ""  # 患者基本信息（脱敏后）
    chief_complaint: str = ""
    present_illness: str = ""
    past_history: str = ""
    personal_history: str = ""
    family_history: str = ""
    physical_exam: str = ""
    auxiliary_exam: str = ""
    preliminary_diagnosis: List[str] = field(default_factory=list)
    treatment_plan: str = ""
    record_id: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if not self.record_id:
            self.record_id = self._generate_record_id()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def _generate_record_id(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_str = hashlib.md5(f"{timestamp}{self.record_type.value}".encode()).hexdigest()[:8]
        return f"MREC-{timestamp}-{hash_str}"


class InputTypeClassifier:
    """输入类型分类器 - 区分对话模式、数据模式、混合模式（加权算法）"""
    
    WEIGHTS = {
        'symptom': 3,    # 症状关键词权重最高
        'data': 2,       # 数据关键词权重
        'history': 2,    # 既往史关键词权重
        'narrative': 1   # 叙事结构关键词权重
    }
    
    SYMPTOM_MARKERS = ['头痛', '胸痛', '腹痛', '发热', '咳嗽', '头晕', '乏力', '恶心', 
                   '呕吐', '呼吸困难', '心悸', '腹泻', '便秘', '关节痛', '腰痛',
                   '视力模糊', '听力下降', '鼻出血', '牙龈出血', '皮肤瘙痒', '水肿',
                   '麻木', '抽搐', '昏迷', '意识模糊', '食欲不振', '体重下降', '口干',
                   '口渴', '多饮', '多尿', '失眠', '记忆减退', '情绪低落', '胸闷',
                   '气短', '腹胀', '黄疸', '皮疹', '寒战', '出汗', '面色苍白', '出冷汗',
                   '关节肿痛', '疼痛', '肿胀', '活动受限', '摔伤', '外伤', '骨折',
                   # 影像学/检查发现（使 mixed 判定更准确）
                   '占位', '结节', '肿块', '阴影', '斑片影', '斑片状', '点状',
                   '条索', '高密度', '低密度', '点片状', '絮状',
                   '反跳痛', '麦氏点', '压痛', '肌紧张', '肠鸣音',
                   '呼之', '睁眼', '对答', '言语不清', '偏瘫', '截瘫',
                   '胎动', '宫缩', '破水', '见红',
                   # 口语化症状变体（覆盖 CC103/CC106 等）
                   '咳', '烧', '发烧', '头疼', '胸口痛', '胸口疼', '肚子疼',
                   '拉肚子', '冒汗', '心慌', '不得劲', '没劲', '浑身',
                   # 口语化症状变体（避免因用词不同被误判为data）
                   '腰疼', '腿疼', '脚疼', '手疼', '背疼',
                   '胃疼', '牙疼', '耳朵疼', '嗓子疼',
                   '眼肿', '脸肿', '脚肿', '腿肿', '手肿',
                   '尿少', '尿多', '尿频', '尿急', '尿痛',
                   '多饮', '多食', '消瘦', '没精神', '睡不好',
                   '吃不下', '便血', '黑便', '咳血', '吐血',
                   '打嗝', '反胃', '烧心', '嗳气',
                   '喘不过气', '上不来气', '憋气', '喘不上气',]
    
    DATA_MARKERS = ['血糖', '血压', 'mmol', 'HbA1c', 'mmHg', '检查示', '化验',
                    '空腹', '餐后', '饭后', '糖化', '尿糖', '血脂', '心电图', 'CT', 'MRI',
                    '胸片', 'B超', '彩超', '血常规', '尿常规', '大便常规',
                    '生化', 'ALT', 'AST', 'TBIL', 'ALB', 'WBC', 'RBC', 'PLT',
                    'HBsAg', 'HBeAg', 'HBcAb', 'U/L', 'μmol', 'g/L',
                    '乙肝', '两对半', '阳性', '阴性',
                    '肝功', '肾功', '电解质', '同型', 'BNP', 'CRP', 'PCT',
                    '尿蛋白', '尿糖', '尿酮体', '便潜血',
                    '体温', 'T', '心率', '脉搏', '血压',
                    'TC', 'TG', 'LDL', 'HDL', 'LDL-C', 'HDL-C',
                    'Cr', 'BUN', 'UA', '尿素氮', '肌酐', '尿酸',
                    'TSH', 'FT3', 'FT4', 'TRAb', 'TPOAb', 'TgAb',
                    'AFP', 'CEA', 'CA199', 'CA153', 'CA125', 'CA724', 'CA50',
                    'PT', 'APTT', 'FIB', 'D-二聚体', 'D二聚体', 'TT',
                    '血气', 'pH', 'PaCO2', 'PaO2', 'HCO3', 'BE', 'Lac', '乳酸',
                    'NC', 'MCV', 'MCH', 'MCHC', 'RDW', 'MPV', 'PDW',
                    'TBIL', 'DBIL', 'IBIL', 'TP', 'GLB', 'A/G',
                    '尿素', '二氧化碳结合力',
                    '白细胞', '中性粒', '淋巴', '嗜酸', '嗜碱', '单核',
                    '血红蛋白', '血小板', '红细胞', '红细胞压积',
                    '空腹血糖', '餐后血糖', '随机血糖',
                    '肿瘤标志物', '甲功', '凝血', 'D-二聚体', '降钙素原',
                    '肝功能', '肾功能', '血脂全套', '心肌酶',
                    '乙肝五项', '乙肝两对半',
                    '诊为', '确诊', '诊断', '淀粉酶']
    
    # 疾病名称（含数据标记子串，需排除）
    DISEASE_EXCLUSIONS = {
        '血压': ['高血压', '低血压'],
        '血糖': ['低血糖'],
    }
    
    HISTORY_MARKERS = ['既往', '病史', '曾患', '以前', '去年', '前年', '多年']
    
    NARRATIVE_MARKERS = ['主诉', '患者', '自觉', '感觉', '不舒服', '疼痛', '伴',
                         '入院', '就诊', '来看病', '说', '诉', '因', '发现']
    
    DIAGNOSIS_MARKERS = ['诊断', '考虑', '疑似', '诊为', '确诊']
    
    @classmethod
    def classify(cls, text: str) -> str:
        """分类输入类型（强制规则 + 加权算法）"""
        text_stripped = text.strip()
        
        # === 强制规则1：随访固定开头优先 ===
        followup_prefixes = ['冠心病随访', '慢阻肺随访', '脑卒中随访', '肾病随访',
                             '甲亢随访', '肿瘤随访', '类风湿随访', '糖尿病随访',
                             '高血压随访', '乙肝随访']
        if any(text_stripped.startswith(p) for p in followup_prefixes):
            return 'mixed'  # 随访 = 混合模式（有症状+有数据）
        # === 强制规则1b：对话标记优先检测 ===
        # 显式对话标记（医/患/家属/护士/病人）→ 无条件对话模式
        # 优先级高于口语化检测，避免口语+数据标记被误判为data
        if re.search(r'(?:医生|患者|家属|护士|病人)(?!姓名)[^：:]{0,3}[：:：]', text_stripped):
            return 'dialogue'
        
        
        # === 强制规则2：口语化对话识别 ===
        if cls._is_colloquial(text_stripped):
            # 口语化但有数据标记（BNP/CEA等）和症状 → mixed模式
            has_data_markers = any(m in text_stripped for m in cls.DATA_MARKERS)
            has_symptom_markers = any(m in text_stripped for m in cls.SYMPTOM_MARKERS)
            # 口语化但有疾病病史描述（如"高血压20年"）→ mixed模式
            has_disease_history = bool(re.search(r'[\u4e00-\u9fa5]{2,6}(病史|史|年|多年|月)', text_stripped))
            if has_data_markers and (has_symptom_markers or has_disease_history):
                return 'mixed'
            # 口语化但仅有数据标记（体征/检验数据）→ data模式
            if has_data_markers:
                return 'data'
            return 'dialogue'
        
        # === 强制规则2b：隐私/个人身份内容 ===
        # 包含"我叫""我是"等自我介绍短语 → 对话
        if re.search(r'(?:我叫|我是|我老婆叫|我爸叫|我妈叫|我丈夫叫|我妻子叫|患者姓名)', text_stripped):
            return 'dialogue'
        # 包含自我介绍"叫" + 电话/身份证 → 对话（即使是"姓名：XXX，电话XXX"）
        if '叫' in text_stripped and re.search(r'(?:电话|手机|身份证)', text_stripped):
            return 'dialogue'
        # 包含"我" + 亲属关系 → 对话
        if re.search(r'我(?:爸|妈|爷|奶|姥|丈|老婆|老公|儿子|女儿|孩子|孙子|孙女|外甥|侄子|哥哥|姐姐|弟弟|妹妹)', text_stripped):
            return 'dialogue'
        # 纯隐私内容（电话/身份证/地址/联系人）且无医学数据 → 对话
        if re.search(r'(?:手机|座机|电话|身份证|身份证号|住址|家庭住址|联系人|家住)', text_stripped):
            has_medical_data = any(m in text_stripped for m in cls.DATA_MARKERS) or any(m in text_stripped for m in cls.SYMPTOM_MARKERS)
            if not has_medical_data:
                return 'dialogue'
        # 纯对话无医学内容但有医疗认知词 → 对话
        if re.search(r'(?:危害性|病的|致病|发病|得病)', text_stripped):
            if not any(m in text_stripped for m in cls.DATA_MARKERS):
                return 'dialogue'
        
        # === 强制规则2c：检验报告/检查结果模式 ===
        # 包含AFP/CEA/CA199等肿瘤标志物+数值 → 随访/复诊
        if re.search(r'(AFP|CEA|CA199|CA153|血常规|生化|凝血|肿瘤标志物).{0,10}?\d+', text_stripped):
            if not cls._is_colloquial(text_stripped) and not re.search(r'(怎么|为什么|是不是|哪里不舒服)', text_stripped):
                return 'mixed'
        # 影像学检查描述（斑片状阴影/结节/占位等）→ 混合模式
        if re.search(r'(斑片|模糊影|阴影|结节|占位|渗出|纤维化|钙化|密度增高|透亮度)', text_stripped):
            if not cls._is_colloquial(text_stripped):
                return 'mixed'
        
        # === 强制规则2d：仅描述症状无既往史 → 对话 ===
        if not any(k in text_stripped for k in ["复查", "复诊", "上次", "之前", "药", "吃完", "随访"]):
            if re.search(r'(怎么|什么|为什么|是不是|哪里不舒服|怎么回事)', text_stripped):
                return 'dialogue'
        
        # === 强制规则3：纯检验数据前缀（仅当整个文本无患者叙述且无症状时）===
        pure_data_prefixes = [
            '肿瘤标志物', '凝血功能', '血气分析', '生化全套', '血脂',
            '肾功能', '甲状腺功能', '心肌酶谱', '血常规', '尿常规',
            '肝功能', '电解质', '糖化血红蛋白', '血糖', '血压',]

        if any(text_stripped.startswith(p) for p in pure_data_prefixes):
            # 检查是否有患者叙述内容
            narrative_indicators = ['患者', '男', '女', '岁', '说', '自觉', '近日', '今日', '复查', '体检']
            has_narrative = any(ind in text_stripped for ind in narrative_indicators)
            # 检查是否有症状标记（影像学发现、占位等）
            has_symptom_marker = any(m in text_stripped for m in cls.SYMPTOM_MARKERS)
            if has_symptom_marker:
                return 'mixed'
            if not has_narrative:
                return 'data'
        
        # === 强制规则4：慢病随访/复诊/复查（有病史+检查数据>对话模式）===
        if re.search(r'(?:高血压|糖尿病|冠心病|乙肝|术后).{0,5}(?:病史|史|年|随访|复查|复诊)', text_stripped):
            return 'mixed'
        if re.search(r'(?:复查|复诊|随访|回来看)', text_stripped) and any(m in text_stripped for m in cls.DATA_MARKERS):
            return 'mixed'
        
        # === 强制规则4b：多系统症状+测量体温→混合模式（DIAG116等）===
        if re.search(r'发热\s*\d+\s*℃', text_stripped):
            symptom_count = sum(1 for m in cls.SYMPTOM_MARKERS if m in text_stripped)
            if symptom_count >= 3:
                return 'mixed'
        
        # === 强制规则5：仅含正常生命体征的头痛/头晕叙述→对话模式===
        # 如果仅有血压/体温/心率且为正常范围，不作为数据标记
        normal_vitals = re.findall(r'血压\s*(\d+)/(\d+)', text_stripped)
        if normal_vitals:
            systolic = int(normal_vitals[0][0])
            diastolic = int(normal_vitals[0][1])
            # 正常血压范围：90-139/60-89
            if 90 <= systolic <= 139 and 60 <= diastolic <= 89:
                # 从 DATA_MARKERS 中去掉血压权重
                pass  # 在加权算法中处理
        
        # === 原有加权算法 ===
        # 预处理：排除疾病名称中的子串误匹配
        clean_text = text
        for marker, diseases in cls.DISEASE_EXCLUSIONS.items():
            if marker in text:
                for disease in diseases:
                    clean_text = clean_text.replace(disease, '')
        
        # 预处理：排除正常血压值（不触发数据模式）
        clean_text_for_data = clean_text
        bp_match = re.search(r'血压\s*(\d+)/(\d+)', text)
        if bp_match:
            systolic = int(bp_match.group(1))
            diastolic = int(bp_match.group(2))
            if 90 <= systolic <= 139 and 60 <= diastolic <= 89:
                clean_text_for_data = clean_text_for_data.replace('血压', '_BP_EXCLUDED_')
        
        # 预处理：排除否定症状（"无XX"中的XX不计入症状得分）
        clean_text_for_symptom = clean_text
        negated_patterns = [
            r'无\s*(\w{1,6})',
            r'没有\s*(\w{1,6})',
            r'否认\s*(\w{1,6})',]

        for pat in negated_patterns:
            for m in re.finditer(pat, text):
                negated_word = m.group(1)
                if negated_word in clean_text_for_symptom:
                    clean_text_for_symptom = clean_text_for_symptom.replace(negated_word, '')
        
        score_symptom = sum(1 for m in cls.SYMPTOM_MARKERS if m in clean_text_for_symptom) * cls.WEIGHTS['symptom']
        score_data = sum(1 for m in cls.DATA_MARKERS if m in clean_text_for_data) * cls.WEIGHTS['data']
        score_history = sum(1 for m in cls.HISTORY_MARKERS if m in text) * cls.WEIGHTS['history']
        score_narrative = sum(1 for m in cls.NARRATIVE_MARKERS if m in text) * cls.WEIGHTS['narrative']
        
        # 诊断关键词也计入叙事分数
        score_narrative += sum(1 for m in cls.DIAGNOSIS_MARKERS if m in text) * cls.WEIGHTS['narrative']
        
        total_score = score_symptom + score_data + score_history + score_narrative
        
        # 判断逻辑
        has_symptom = score_symptom > 0
        has_data = score_data > 0
        has_history = score_history > 0
        has_diagnosis = any(m in text for m in cls.DIAGNOSIS_MARKERS)
        
        # 混合模式判定：含症状 + 数据标记（既往史不单独触发混合）
        if has_symptom and has_data:
            # 只有当数据量极少且症状/叙事远超数据时才判为对话
            if score_data < 2 and (score_narrative + score_symptom) > score_data * 10:
                return "dialogue"
            return "mixed"
        # 诊断模式：含诊断关键词
        elif has_diagnosis:
            return "dialogue"
        # 纯对话模式：只有症状或叙事结构
        elif has_symptom or score_narrative > 0:
            # 如果叙事结构+数据标记并存，判为混合而非纯对话
            if has_data:
                # 但若数据特征远少于对话特征，仍判为对话
                if score_narrative + score_symptom > score_data * 3:
                    return "dialogue"
                return "mixed"
            return "dialogue"
        # 纯数据模式：只有数据关键词
        elif has_data or has_history:
            return "data"
        else:
            return "unknown"
    
    @staticmethod
    def _is_colloquial(text: str) -> bool:
        """判断是否为口语化对话输入"""
        indicators = [
            "好几天了", "一直", "这两天", "摔了一跤", "摸着热",
            "隐隐", "有点", "感觉", "怎么办", "医生", "老张", "老王",
            "说", "不得劲", "不舒服", "吃饭不香", "睡觉不好",
            "就是那个", "我妈", "他爸", "孩子", "抱着", "哭着",
            "浑身", "没劲", "疼得厉害", "站不起来", "动不了",
            "大夫", "一个星期", "一个月", "冒汗", "胸口",
            "三个小时", "两个小时", "好几天",]

        score = sum(1 for w in indicators if w in text)
        # 短文本 + 高口语词密度
        return len(text) < 150 and score >= 2


class LanguageDetector:
    """语言检测器 - 检测输入语言并拦截非中文输入及中英文混杂输入"""
    
    CHINESE_CHARS = '的一是了我有他在人这上们来到时大地为子中你说生国年着那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感'
    CHINESE_SYMBOLS = '，。！？；：、（）【】《》''""'
    
    # 医学术语缩写白名单（不计入英文单词计数）
    MEDICAL_ABBREVIATIONS = {
        'ALT', 'AST', 'TBIL', 'ALB', 'WBC', 'RBC', 'PLT', 'HbA1c',
        'HBsAg', 'HBeAg', 'HBcAb', 'BNP', 'CRP', 'PCT', 'FBG',
        'CT', 'MRI', 'DR', 'ECG', 'EEG',
        'mmol', 'mmHg', 'mg', 'g', 'L', 'ml', 'kg', 'cm',
        # 血脂
        'TC', 'TG', 'LDL', 'HDL', 'LDL-C', 'HDL-C',
        # 肾功能
        'Cr', 'BUN', 'UA',
        # 肝功能
        'DBIL', 'IBIL', 'TP', 'GLB',
        # 甲状腺功能
        'TSH', 'FT3', 'FT4', 'TRAb', 'TPOAb', 'TgAb',
        # 肿瘤标志物
        'AFP', 'CEA', 'CA199', 'CA153', 'CA125', 'CA724', 'CA50',
        # 凝血功能
        'PT', 'APTT', 'FIB', 'TT',
        # 血气分析
        'PaCO2', 'PaO2', 'PCO2', 'PO2', 'HCO3', 'BE', 'Lac', 'pH',
        # 血常规
        'Hb', 'MCV', 'MCH', 'MCHC', 'RDW', 'MPV', 'PDW',
        # 心功能
        'NT-proBNP', 'proBNP',
        # 单位
        'mIU', 'pmol', 'μmol', 'ng', 'U', 's',
        # 用药频率 / 给药途径 / 临床缩写
        'EF', 'qd', 'bid', 'tid', 'qid', 'qn', 'prn', 'po', 'iv', 'im', 'sc', 'pg',
        'NT', 'NIHSS', 'BP', 'ST',
    }
    @classmethod
    def count_english_words(cls, text: str) -> int:
        """统计英文单词数（排除医学术语缩写）"""
        # 预处理：拆分数字+单位（如20mg→20 mg），拆分连字符术语（如NT-proBNP→NT proBNP），拆分字母+数字（如T36→T 36）
        text = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', text)
        text = re.sub(r'([A-Z]{2,})-([a-zA-Z]+)', r'\1 \2', text)
        text = re.sub(r'([A-Za-z])(\d+)', r'\1 \2', text)
        # 匹配连续的字母数字组合（至少2个字母）
        words = re.findall(r'[a-zA-Z][a-zA-Z0-9]{1,}', text)
        # 过滤掉医学术语缩写（不区分大小写）
        medical_lower = {w.lower() for w in cls.MEDICAL_ABBREVIATIONS}
        non_medical = [w for w in words if w.lower() not in medical_lower]
        return len(non_medical)
    
    @classmethod
    def is_chinese(cls, text: str, threshold: float = 0.3) -> bool:
        """检测是否为中文文本"""
        text = text.strip()
        if not text:
            return True  # 空文本视为中文
        
        # 检查是否包含中文汉字或中文标点
        has_chinese_char = any(c in cls.CHINESE_CHARS for c in text)
        has_chinese_symbol = any(c in cls.CHINESE_SYMBOLS for c in text)
        has_chinese_range = any('\u4e00' <= c <= '\u9fff' for c in text)
        
        # 如果有任何中文特征，视为中文
        if has_chinese_char or has_chinese_symbol or has_chinese_range:
            return True
        
        # 如果没有任何中文特征，检查是否全是英文字母
        is_all_english = text.replace(' ', '').replace(',', '').replace('.', '').isalpha()
        if is_all_english:
            return False
        
        # 其他情况（如乱码、数字等），视为中文（让后续处理）
        return True
    
    @classmethod
    def detect(cls, text: str) -> str:
        """检测语言类型"""
        if not cls.is_chinese(text):
            return "english"
        return "chinese"
    
    @classmethod
    def detect_mixed(cls, text: str) -> Tuple[str, Optional[str]]:
        """检测是否存在中英文混杂问题
        Returns:
            (language, error_message)
            - ("chinese", None): 纯中文或可接受的中文为主
            - ("english", "暂不支持英文输入..."): 英文为主
            - ("mixed", "输入包含过多英文..."): 中英文混杂
        """
        if not cls.is_chinese(text):
            return "english", "暂不支持英文输入，请使用中文描述患者情况"
        
        english_words = cls.count_english_words(text)
        # 医学缩写不计入英文词数（LAD, PCI, qd, bid等）
        med_abbr_pattern = re.compile(r'(?:LAD|RCA|LCX|PCI|CABG|STEMI|NSTEMI|CT|MRI|PET|ECG|EKG|EEG|'
                                     r'BNP|NT-proBNP|CRP|PCT|ESR|WBC|RBC|Hb|HCT|PLT|PT|APTT|INR|'
                                     r'ALT|AST|TBIL|DBIL|ALB|BUN|Cr|GFR|CCr|'
                                     r'qd|bid|tid|qid|qn|prn|po|iv|im|sc|'
                                     r'FEV1|FVC|PEF|SpO2|PaO2|PaCO2|pH|'
                                     r'HbA1c|TSH|FT3|FT4|CEA|AFP|CA199|CA125|CA153|PSA|'
                                     r'BMI|BSA|DOB|y/o|yo|cm|mm|kg|mg|g|ml|L|U|mmol|'
                                     r'COPD|ARDS|DIC|SLE|RA|IBD|GERD|'
                                     r'PTCA|DSA|MRCP|ERCP|TACE|HBsAg|HBeAg|HBcAb|'
                                     r'DNA|RNA|PCR|ELISA)', re.IGNORECASE)
        med_abbr_count = len(med_abbr_pattern.findall(text))
        effective_english = english_words - med_abbr_count
        if effective_english > 4:
            return "english", f"输入包含过多英文词汇（{english_words}个），请使用中文描述患者情况"
        
        return "chinese", None


class PrivacyDesensitizer:
    """隐私脱敏器 - 先提取到结构化字段，再对字段值脱敏，避免全文误伤"""
    
    PROTECTED_WORDS = ['待补充', '患者基本信息', '患者病情', '患者情况', '患者病*', '患者情*',
                       '意识', '知识', '常识', '见识', '认识', '知道', '识别', '认知',
                       '意识模糊', '意识不清', '意识障碍',
                       '症状', '体征', '诊断', '治疗', '疾病', '阳性', '阴性']
    
    @classmethod
    def desensitize_value(cls, value_type: str, value: str) -> str:
        """对字段值进行脱敏（字段级别的脱敏，不涉及全文）"""
        if not value or value == '待补充':
            return value
        
        if value_type == 'name':
            # 姓名：保留首字，其余用*
            # 李小明 → 李**
            # 老张 → 老*
            if len(value) >= 2:
                return value[0] + '*' * (len(value) - 1)
            return value
        
        elif value_type == 'phone':
            # 电话：保留前3后4，中间****
            # 13987654321 → 139****4321
            clean = re.sub(r'\D', '', value)
            if len(clean) == 11:
                return clean[:3] + '****' + clean[-4:]
            return value
        
        elif value_type == 'id_card':
            # 身份证：保留前6（地区码+生日年月），后4，中间8个*
            # 110101199505152345 → 110101********2345
            clean = re.sub(r'\D', '', value)
            if len(clean) == 18:
                return clean[:6] + '*' * 8 + clean[-4:]
            elif len(clean) > 10:
                return clean[:3] + '*' * (len(clean) - 7) + clean[-4:]
            return value
        
        elif value_type == 'address':
            # 地址：保留到区/县，后面****
            pattern = r'((?:[\u4e00-\u9fa5]+省)?(?:[\u4e00-\u9fa5]+市)?(?:[\u4e00-\u9fa5]+区|[\u4e00-\u9fa5]+县))(.+)'
            match = re.search(pattern, value)
            if match:
                return match.group(1) + '****'
            return value
        
        return value
    
    @classmethod
    def desensitize(cls, text: str) -> str:
        """对格式化的输出文本做脱敏处理（仅对结构化字段做脱敏，保护医学术语）
        注意：此方法不应在全文上匹配姓名/电话，只针对格式化输出中的已知字段
        """
        # 先保护医学术语
        for i, word in enumerate(cls.PROTECTED_WORDS):
            text = text.replace(word, f'__PROTECTED_{i}__')
        
        # 在格式化输出中，患者信息已经通过字段脱敏，但可能还有：
        # 1. 独立电话号码（不在格式化字段中）
        text = re.sub(r'(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)', r'\1****\2', text)
        # 1b. 座机号码（0xxx-xxxxxxxx，保留区号+后4位，中间****）
        text = re.sub(r'(?<!\d)(0\d{2,3})[-.]?(\d{4})(\d{3,4})(?!\d)', r'\1****\3', text)
        # 2. 独立身份证号（18位）  
        text = re.sub(r'(?<!\d)(\d{6})\d{8}(\d{4})(?!\d)', r'\1********\2', text)
        
        # 恢复保护词
        for i, word in enumerate(cls.PROTECTED_WORDS):
            text = text.replace(f'__PROTECTED_{i}__', word)
        
        return text
    
    @classmethod
    def desensitize_all(cls, text: str) -> str:
        """全面脱敏（同desensitize，保持一致）"""
        return cls.desensitize(text)


class HallucinationChecker:
    """幻觉检测器 - 检查输出是否包含输入中不存在的信息"""
    
    @classmethod
    def extract_numbers(cls, text: str) -> set:
        """提取文本中的所有数字"""
        numbers = set()
        # 提取血压值
        bp_matches = re.findall(r'(\d{2,3})/(\d{2,3})', text)
        for match in bp_matches:
            numbers.update(match)
        
        # 提取血糖、HbA1c等数值
        num_matches = re.findall(r'(\d+\.?\d*)', text)
        numbers.update(num_matches)
        
        return numbers
    
    @classmethod
    def extract_dates(cls, text: str) -> set:
        """提取文本中的日期"""
        dates = set()
        # 匹配日期格式
        date_patterns = [
            r'\d{4}年\d{1,2}月\d{1,2}日',
            r'\d{1,2}月\d{1,2}日',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}/\d{1,2}/\d{4}']

        for pattern in date_patterns:
            dates.update(re.findall(pattern, text))
        return dates
    
    @classmethod
    def check(cls, output: str, input_text: str) -> Tuple[bool, List[str]]:
        """检查输出是否有幻觉"""
        issues = []
        
        # 检查日期幻觉
        input_dates = cls.extract_dates(input_text)
        output_dates = cls.extract_dates(output)
        if output_dates and not input_dates:
            issues.append(f"日期幻觉：输出包含日期{output_dates}，但输入中没有日期")
        
        # 检查数值幻觉（主要针对医疗指标）
        input_nums = cls.extract_numbers(input_text)
        output_nums = cls.extract_numbers(output)
        
        # 检查输出中的数值是否都能在输入中找到来源
        # 允许一些常见的默认数值（如体温37、心率72等）
        safe_defaults = {'37', '72', '120', '80', '60', '16', '20', '100'}
        
        for num in output_nums:
            # 跳过安全默认值
            if num in safe_defaults:
                continue
            # 检查该数值是否在输入中存在
            if num not in input_nums:
                # 检查是否是输入数值的合理转换（如7.8和78）
                found = False
                for input_num in input_nums:
                    if num in input_num or input_num in num:
                        found = True
                        break
                if not found:
                    issues.append(f"数值幻觉：输出包含数值{num}，但输入中无来源")
        
        return len(issues) == 0, issues


class DirectionGuard:
    """诊断方向校验器 - 确保诊断方向与症状系统一致，防止跨系统误判"""
    
    # 症状关键词 → 所属系统
    SYMPTOM_SYSTEM = {
        # 呼吸系统
        "咳嗽": "呼吸", "咳痰": "呼吸", "喘息": "呼吸", "气促": "呼吸",
        "胸闷": "呼吸", "胸痛": "呼吸", "咽痛": "ENT", "声嘶": "ENT",
        "鼻塞": "ENT", "流涕": "ENT", "喉咙": "ENT", "咽喉": "ENT",
        "发热": "*",  # 通配，不做限制
        
        # 心血管
        "心悸": "心血管", "心慌": "心血管", "水肿": "心血管",
        "呼吸困难": "心血管", "端坐": "心血管",
        
        # 消化
        "腹痛": "消化", "腹泻": "消化", "恶心": "消化", "呕吐": "消化",
        "腹胀": "消化", "反酸": "消化", "嗳气": "消化", "便秘": "消化",
        "便血": "消化", "黄疸": "消化", "食欲": "消化",
        
        # 骨骼肌肉
        "腰痛": "骨科", "背痛": "骨科", "关节": "骨科", "膝": "骨科",
        "肩": "骨科", "颈": "骨科", "腿痛": "骨科", "腿麻": "骨科",
        "四肢": "骨科", "骨折": "骨科", "外伤": "骨科", "扭伤": "骨科",
        "畸形": "骨科",
        
        # 神经
        "头痛": "神经", "头晕": "神经", "眩晕": "神经", "意识": "神经",
        "抽搐": "神经", "麻木": "神经", "失眠": "精神",
        
        # 泌尿/肾
        "血尿": "泌尿", "尿频": "泌尿", "尿急": "泌尿", "尿痛": "泌尿",
        "泡沫尿": "泌尿", "排尿": "泌尿",
        
        # 妇科
        "月经": "妇科", "阴道": "妇科", "同房": "妇科", "停经": "妇科",
        "白带": "妇科",
        
        # 皮肤
        "皮疹": "皮肤", "瘙痒": "皮肤", "风团": "皮肤", "红斑": "皮肤",
        "出血点": "皮肤", "瘀点": "皮肤", "瘀斑": "皮肤",
        
        # 眼科
        "眼": "眼科", "视力": "眼科",
        
        # 耳科
        "耳鸣": "ENT", "耳闷": "ENT", "听力": "ENT",
        
        # 内分泌
        "多饮": "内分泌", "多尿": "内分泌", "口渴": "内分泌", "消瘦": "内分泌",
    }
    
    # 诊断关键词 → 所属系统
    DIAG_SYSTEM = {
        "肺炎": "呼吸", "支气管炎": "呼吸", "上呼吸道": "呼吸",
        "COPD": "呼吸", "慢阻肺": "呼吸", "哮喘": "呼吸",
        "肺结核": "呼吸", "胸膜炎": "呼吸", "气胸": "呼吸",
        
        "心力衰竭": "心血管", "心衰": "心血管", "冠心病": "心血管",
        "心肌梗死": "心血管", "心梗": "心血管", "心律失常": "心血管",
        "房颤": "心血管", "高血压": "心血管", "心绞痛": "心血管",
        "急性冠脉": "心血管",
        
        "阑尾炎": "消化", "胃肠炎": "消化", "胆囊炎": "消化",
        "胰腺炎": "消化", "肝炎": "消化", "肝硬化": "消化",
        "胃溃疡": "消化", "十二指肠": "消化", "肠梗阻": "消化",
        "消化不良": "消化", "胃食管": "消化",
        
        "骨折": "骨科", "韧带": "骨科", "间盘": "骨科",
        "颈椎": "骨科", "腰椎": "骨科", "关节炎": "骨科",
        "软组织": "骨科",
        
        "脑梗死": "神经", "脑出血": "神经", "癫痫": "神经",
        "偏头痛": "神经", "蛛网膜": "神经", "帕金森": "神经",
        "阿尔茨海默": "神经",
        
        "抑郁": "精神", "焦虑": "精神", "躯体化": "精神",
        
        "肾炎": "泌尿", "肾病": "泌尿", "肾结石": "泌尿",
        "泌尿道": "泌尿", "尿路": "泌尿",
        
        "早孕": "妇科", "妊娠": "妇科", "流产": "妇科",
        "卵巢": "妇科", "宫颈": "妇科", "月经": "妇科",
        "多囊": "妇科",
        
        "皮炎": "皮肤", "湿疹": "皮肤", "痤疮": "皮肤",
        "荨麻疹": "皮肤", "紫癜": "皮肤",
        
        "中耳炎": "ENT", "喉炎": "ENT", "扁桃体": "ENT",
        "声带": "ENT", "鼻窦炎": "ENT", "咽炎": "ENT",
        
        "糖尿病": "内分泌", "甲亢": "内分泌", "甲减": "内分泌",
        "甲状腺": "内分泌",
        
        "肿瘤": "*", "癌症": "*", "癌": "*", "感染": "*",
        "待补充": "*", "待查": "*", "可能": "*",
    }
    
    @classmethod
    def check(cls, chief_complaint: str, diagnosis: str, raw_input: str) -> str:
        """检测诊断方向是否与主诉一致。返回空字符串=无问题，否则返回警告。
        
        仅做硬性跨系统检测（如腰痛→肺炎），不做细粒度判断。
        """
        if not chief_complaint or not diagnosis:
            return ""
        
        # 从主诉+原文中提取症状系统
        complaint_systems = set()
        for symptom, system in cls.SYMPTOM_SYSTEM.items():
            if symptom in chief_complaint or symptom in raw_input:
                complaint_systems.add(system)
        
        # 从诊断中提取诊断系统
        diag_systems = set()
        for diag_key, system in cls.DIAG_SYSTEM.items():
            if diag_key in diagnosis:
                diag_systems.add(system)
        
        # 通配符介入
        complaint_systems.discard("*")
        diag_systems.discard("*")
        
        # 两个方向都没有或任一方未提取到系统 → 不做限制
        if not complaint_systems or not diag_systems:
            return ""
        
        # 有交集 → 合理
        if complaint_systems & diag_systems:
            return ""
        
        # 无交集 → 标记警告
        logger.warning(f"方向不一致: 症状系统={complaint_systems}, 诊断系统={diag_systems}, CC={chief_complaint[:20]}, Diag={diagnosis[:30]}")
        return f"诊断方向与主诉不匹配（症状={complaint_systems}，诊断={diag_systems}）"


class MedicalValueValidator:
    """医学数值校验器 - 校验医学数值的合理性"""
    
    VALID_RANGES = {
        'temperature': (30.0, 45.0),      # ℃
        'heart_rate': (30, 250),          # 次/分
        'respiratory_rate': (8, 60),      # 次/分
        'blood_pressure_sys': (50, 300),  # mmHg
        'blood_pressure_dia': (30, 200),  # mmHg
        'blood_glucose': (0.5, 50.0),     # mmol/L
        'age': (0, 150),                  # 岁
        'hba1c': (2.0, 20.0),            # %
    }
    
    @classmethod
    def validate(cls, value_type: str, value: float) -> Tuple[bool, Optional[str]]:
        """校验单个医学数值
        Returns:
            (is_valid, error_message)
        """
        if value_type not in cls.VALID_RANGES:
            return True, None
        
        min_val, max_val = cls.VALID_RANGES[value_type]
        if value < min_val or value > max_val:
            return False, f"数值{value}超出正常范围({min_val}-{max_val})，请核实"
        
        return True, None
    
    @classmethod
    def validate_temperature(cls, val: float) -> Tuple[str, str]:
        """校验体温并返回状态和描述"""
        if val < 30 or val > 45:
            return 'ABNORMAL', f'体温{val}℃超出人体可能范围，请核实'
        elif val < 36:
            return 'LOW', f'体温{val}℃（偏低）'
        elif val > 37.3:
            return 'HIGH', f'体温{val}℃（发热）'
        else:
            return 'NORMAL', f'体温{val}℃（正常）'

    @classmethod
    def validate_blood_pressure(cls, sys: float, dia: float) -> Tuple[str, str]:
        """校验血压并返回状态和描述"""
        if sys <= 0 or dia <= 0:
            return 'INVALID', f'血压{sys}/{dia} mmHg（数值不可能为0或负数）'
        if sys > 300 or dia > 200:
            return 'INVALID', f'血压{sys}/{dia} mmHg（超出人体极限）'
        if sys < 90 or dia < 60:
            return 'LOW', f'血压{sys}/{dia} mmHg（低血压）'
        if sys >= 140 or dia >= 90:
            return 'HIGH', f'血压{sys}/{dia} mmHg（高血压）'
        return 'NORMAL', f'血压{sys}/{dia} mmHg（正常）'

    @classmethod
    def validate_blood_glucose(cls, val: float) -> Tuple[str, str]:
        """校验血糖并返回状态和描述"""
        if val < 0.5 or val > 50:
            return 'ABNORMAL', f'血糖{val} mmol/L（超出人体可能范围，请核实）'
        elif val < 3.9:
            return 'LOW', f'血糖{val} mmol/L（低血糖）'
        elif val > 6.1:
            return 'HIGH', f'血糖{val} mmol/L（高血糖）'
        else:
            return 'NORMAL', f'血糖{val} mmol/L（正常）'

    @classmethod
    def extract_all_temperatures(cls, text: str) -> List[dict]:
        """提取文本中所有体温值并校验"""
        results = []
        # 匹配 体温数字度/℃ 或 T数字度/℃
        for match in re.finditer(r'(?:体温|T)[\s：:]*(-?\d+\.?\d*)\s*[度℃°]?', text):
            try:
                val = float(match.group(1))
                status, desc = cls.validate_temperature(val)
                results.append({'value': val, 'status': status, 'description': desc})
            except ValueError:
                pass
        
        return results

    @classmethod
    def extract_all_blood_pressures(cls, text: str) -> List[dict]:
        """提取文本中所有血压值并校验"""
        results = []
        for match in re.finditer(r'(?:血压)[\s：:]*(\d{1,3})\s*/\s*(\d{1,3})', text):
            try:
                sys_val = int(match.group(1))
                dia_val = int(match.group(2))
                status, desc = cls.validate_blood_pressure(sys_val, dia_val)
                results.append({'sys': sys_val, 'dia': dia_val, 'status': status, 'description': desc})
            except ValueError:
                pass
        return results

    @classmethod
    def extract_all_blood_sugars(cls, text: str) -> List[dict]:
        """提取文本中所有血糖值并校验"""
        results = []
        for match in re.finditer(r'(?:空腹)?血糖[\s：:]*(-?\d+\.?\d*)', text):
            try:
                val = float(match.group(1))
                status, desc = cls.validate_blood_glucose(val)
                results.append({'value': val, 'status': status, 'description': desc})
            except ValueError:
                pass
        return results

    @classmethod
    def format_temperatures(cls, results: List[dict]) -> str:
        """格式化多体温值为显示字符串"""
        if not results:
            return ""
        parts = []
        abnormal_parts = []
        for r in results:
            val_str = f"T{r['value']}℃"
            if r['status'] == 'ABNORMAL':
                abnormal_parts.append(f"{val_str}超出范围")
            elif r['status'] == 'LOW':
                parts.append(f"{val_str}（偏低）")
            elif r['status'] == 'HIGH':
                parts.append(f"{val_str}（发热）")
            else:
                parts.append(val_str)
        result = "，".join(parts) if parts else ""
        if abnormal_parts:
            result += f"[异常：{'，'.join(abnormal_parts)}]"
        return result

    @classmethod
    def format_blood_pressures(cls, results: List[dict]) -> str:
        """格式化多血压值为显示字符串"""
        if not results:
            return ""
        parts = []
        abnormal_parts = []
        for r in results:
            val_str = f"BP{r['sys']}/{r['dia']} mmHg"
            if r['status'] in ('INVALID',):
                abnormal_parts.append(f"{val_str}超出范围")
            elif r['status'] == 'LOW':
                parts.append(f"{val_str}（低血压）")
            elif r['status'] == 'HIGH':
                parts.append(f"{val_str}（高血压）")
            else:
                parts.append(val_str)
        result = "，".join(parts) if parts else ""
        if abnormal_parts:
            result += f"[异常：{'，'.join(abnormal_parts)}]"
        return result

    @classmethod
    def format_blood_sugars(cls, results: List[dict]) -> str:
        """格式化多血糖值为显示字符串"""
        if not results:
            return ""
        parts = []
        abnormal_parts = []
        for r in results:
            val_str = f"血糖{r['value']} mmol/L"
            if r['status'] == 'ABNORMAL':
                abnormal_parts.append(f"{val_str}超出范围")
            elif r['status'] == 'LOW':
                parts.append(f"{val_str}（低血糖）")
            elif r['status'] == 'HIGH':
                parts.append(f"{val_str}（高血糖）")
            else:
                parts.append(val_str)
        result = "，".join(parts) if parts else ""
        if abnormal_parts:
            result += f"[异常：{'，'.join(abnormal_parts)}]"
        return result

    @classmethod
    def validate_info(cls, info: ExtractedInfo) -> Tuple[ExtractedInfo, List[str]]:
        """校验提取信息中的所有医学数值
        Returns:
            (validated_info, warning_messages)
        """
        warnings = []
        
        # 校验体温
        if info.temperature:
            try:
                temp_val = float(info.temperature.replace('℃', ''))
                valid, msg = cls.validate('temperature', temp_val)
                if not valid:
                    warnings.append(f"体温{msg}")
                    info.temperature = ""  # 清除异常值
            except ValueError:
                pass
        
        # 校验发热温度
        if info.fever_temp:
            try:
                fever_val = float(info.fever_temp.replace('℃', ''))
                valid, msg = cls.validate('temperature', fever_val)
                if not valid:
                    warnings.append(f"发热{msg}")
                    info.fever_temp = ""
            except ValueError:
                pass
        
        # 校验心率
        if info.heart_rate:
            try:
                hr_val = int(info.heart_rate.replace('次/分', ''))
                valid, msg = cls.validate('heart_rate', hr_val)
                if not valid:
                    warnings.append(f"心率{msg}")
                    info.heart_rate = ""
            except ValueError:
                pass
        
        # 校验血糖
        if info.blood_sugar:
            try:
                bs_val = float(re.search(r'[\d.]+', info.blood_sugar).group())
                valid, msg = cls.validate('blood_glucose', bs_val)
                if not valid:
                    warnings.append(f"血糖{msg}")
                    info.blood_sugar = ""
            except (ValueError, AttributeError):
                pass
        
        # 校验血压
        if info.blood_pressure:
            bp_match = re.search(r'(\d+)/(\d+)', info.blood_pressure)
            if bp_match:
                sys_val, dia_val = int(bp_match.group(1)), int(bp_match.group(2))
                valid_sys, msg_sys = cls.validate('blood_pressure_sys', sys_val)
                valid_dia, msg_dia = cls.validate('blood_pressure_dia', dia_val)
                if not valid_sys:
                    warnings.append(f"收缩压{msg_sys}")
                    info.blood_pressure = ""
                elif not valid_dia:
                    warnings.append(f"舒张压{msg_dia}")
                    info.blood_pressure = ""
        
        return info, warnings


class SuspiciousInputDetector:
    """攻击意图/虚假信息检测器（改进版 - 区分测试意图和医学数值测试）"""
    
    SUSPICIOUS_WORDS = ['哈哈', '测试', '乱写', '假的', '骗人的', '没救了']
    MEDICAL_INTERNAL_WORDS = ['其实我没病']
    
    MEDICAL_INDICATORS = ['血糖', '血压', '体温', '患者', '主诉', '诊断',
                          '症状', '查体', '检验', '检查', '病史', '入院',
                          '发热', '咳嗽', '腹痛', '头痛', '胸痛', '外伤']
    
    @classmethod
    def check(cls, text: str) -> Tuple[bool, Optional[str]]:
        """检测输入是否存在攻击意图
        改进：只有当检测到测试/虚假用词且不含真实医学内容时才拦截
        Returns:
            (is_suspicious, warning_message)
        """
        has_suspicious_word = any(w in text for w in cls.SUSPICIOUS_WORDS)
        has_medical_indicator = any(w in text for w in cls.MEDICAL_INDICATORS)
        
        # "其实我没病"是极强的测试信号，无论是否有医学内容都拦截
        for word in cls.MEDICAL_INTERNAL_WORDS:
            if word in text:
                return True, "检测到疑似测试内容，请提供真实病情描述"
        
        # 普通测试用词 + 无医学内容 → 拦截
        if has_suspicious_word and not has_medical_indicator:
            return True, "检测到疑似测试内容，请提供真实病情描述"
        
        # 即使有测试用词，如果有真实医学内容也放行
        # 纯数值列表（如"血糖3.5，血糖5.6"）是正常医学测试
        return False, None


class MedicalTermTranslator:
    """医学术语翻译器 - 将英文医学术语翻译为中文"""
    
    # 英文→中文映射表
    TERM_MAP = {
        'STEMI': '急性ST段抬高型心肌梗死',
        'NSTEMI': '急性非ST段抬高型心肌梗死',
        'PCI': '经皮冠状动脉介入治疗',
        'CABG': '冠状动脉旁路移植术',
        'COPD': '慢性阻塞性肺疾病',
        'DM': '糖尿病',
        'HTN': '高血压',
        'CAD': '冠状动脉粥样硬化性心脏病',
        'CHD': '冠心病',
        'CHF': '充血性心力衰竭',
        'AF': '心房颤动',
        'PE': '肺栓塞',
        'DVT': '深静脉血栓',
        'ARDS': '急性呼吸窘迫综合征',
        'AMI': '急性心肌梗死',
        'CVA': '脑血管意外',
        'TIA': '短暂性脑缺血发作',
        'UTI': '尿路感染',
        'CAP': '社区获得性肺炎',
        'HAP': '医院获得性肺炎',
        'CKD': '慢性肾脏病',
        'ESRD': '终末期肾病',
        'IBD': '炎症性肠病',
        'SLE': '系统性红斑狼疮',
        'RA': '类风湿关节炎',
        'OA': '骨关节炎',
        'qd': '每日一次',
        'bid': '每日两次',
        'tid': '每日三次',
        'qid': '每日四次',
        'po': '口服',
        'iv': '静脉注射',
        'im': '肌肉注射',
        'ih': '皮下注射',
        'prn': '必要时',
    }
    
    @classmethod
    def translate(cls, text: str) -> str:
        """翻译文本中的英文医学术语为中文"""
        result = text
        for en_term, cn_term in cls.TERM_MAP.items():
            result = re.sub(r'(?<![a-zA-Z])' + re.escape(en_term) + r'(?![a-zA-Z])', cn_term, result, flags=re.IGNORECASE)
        return result


# ======== API 安全套话检测 + 规则兜底 ========
SAFE_PHRASES = [
    "需结合临床", "重复检测确认", "建议进一步检查",
    "未能识别有效医学信息", "信息不足", "无法确定",
    "请提供更多信息", "请补充更详细",
    "请提供患者信息", "请提供更详细的", "无法判断",]


def is_safe_phrase(text: str) -> bool:
    """检测API是否输出安全套话"""
    if not text:
        return True
    # Allow legitimate medical diagnoses containing "待查"/"待排"
    if any(kw in text for kw in ["待查", "待排"]):
        return False
    return any(phrase in text for phrase in SAFE_PHRASES)

# 诊断关键词映射（用于规则兜底）



def build_cc_from_symptoms(pos_symptoms: list, neg_symptoms: list, text: str) -> str:
    """从症状列表构造规则主诉（比LLM更可靠）
    
    策略：从阳性症状中选最重要的1-2个+第一个时间参考，组成规范主诉。
    "重要"定义为：有明确部位的症状 > 全身症状 > 检验异常描述的伪症状
    """
    if not pos_symptoms:
        return ""
    
    # 去除伪症状（来自检验数据的描述性标签）
    pseudo_symptoms = {"血糖异常", "肝功能异常", "肾功能异常", "贫血", "指标异常",
                       "电解质紊乱", "肾功能不全", "心动过速", "心动过缓",
                       "待补充", "检验异常", "影像异常", "正常", "未见异常"}
    
    real_symptoms = [s for s in pos_symptoms if s not in pseudo_symptoms]
    if not real_symptoms:
        return ""
    
    # 去重：移除子串症状（如["牙龈出血", "出血"] -> ["牙龈出血"]）
    deduped = []
    for s in sorted(real_symptoms, key=len, reverse=True):
        if not any(s in other and s != other for other in deduped):
            deduped.append(s)
    # 恢复原始顺序
    real_symptoms = [s for s in real_symptoms if s in deduped]
    
    # 找到与主症状最相关的时间参考（优先短时程）
    time_matches = list(re.finditer(r"(\d+\s*(?:天|周|个月|月|年|小时|分钟|日))", text))
    time_str = ""
    if time_matches:
        for prefer in ['天', '小时', '分钟']:
            for m in time_matches:
                if prefer in m.group(1):
                    time_str = m.group(1)
                    break
            if time_str:
                break
        if not time_str:
            time_str = time_matches[-1].group(1)
    
    # ===== 症状重要性评分（部位>系统>全身>机制描述） =====
    # 高优先级：有明确解剖部位的精确症状
    HIGH_PRIORITY = {'头痛', '胸痛', '腹痛', '腰痛', '关节痛', '咽痛', '耳痛', '眼痛',
                     '牙痛', '颈痛', '肩痛', '背痛', '膝痛', '下腹痛', '上腹痛', '右下腹痛',
                     '咳嗽', '咳黄痰', '咳白痰', '呼吸困难', '心悸',
                     '抽搐', '昏迷', '意识模糊', '意识不清',
                     '鼻出血', '牙龈出血', '便血', '黑便', '咯血', '血尿',
                     '皮疹', '黄疸', '水肿', '双下肢水肿', '下肢水肿',
                     '吞咽困难', '声音嘶哑', '停止排气排便',
                     '外伤', '骨折', '扭伤', '摔伤',
                     '胎动减少', '阴道流血', '停经',
                     '视力模糊', '视物模糊', '听力下降',
                     '胸闷', '胸闷气短', '气短', '喘息',
                     '尿频', '尿急', '尿痛', '肉眼血尿',
                     '鼻塞', '流涕', '喷嚏', '耳鸣', '耳闷',
                     '反酸', '烧心', '腹胀', '打嗝', '嗳气',
                     '气促', '发热伴咳嗽',
                     '发热', '头晕', '眩晕', '膝关节痛'}
    # 中优先级：系统/全身症状，有价值但不够精确
    MED_PRIORITY = {'乏力', '恶心', '失眠', '食欲不振', '体重下降',
                    '口干', '口渴', '多饮', '多尿', '寒战', '面色苍白', '出冷汗',
                    '便秘', '麻木', '皮肤瘙痒', '精神差',
                    '隐痛', '刺痛', '胀痛', '绞痛', '灼痛', '抽痛', '坠痛',
                    '反跳痛', '乳房胀痛', '夜间盗汗', '口干眼干',
                    '水样便', '活动后气促', '活动后胸闷',
                    '剧烈头痛', '压榨性胸痛',
                    '腹部不适',
                    '呼之可睁眼', '对答不清', '嗜睡', '昏睡', '谵妄',
                    '偏瘫', '截瘫', '口角歪斜', '言语不清', '肢体无力',
                    '呕吐', '腹泻'}
    # 低优先级：过于通用，或非症状性描述
    LOW_PRIORITY = {'疼痛', '出血', '肿胀', '活动受限', '有痰', '没劲', '无力',
                    '车祸', '撞伤', '擦伤', '撕裂', '流血', '伤口', '畸形',
                    '吃奶少', '走路没劲', '喘', '干咳',
                    '高血压', '糖尿病', '冠心病', '肝炎', '乙肝', '脂肪肝',
                    '阑尾炎', '胰腺炎', '胃炎', '肺炎', '支气管炎',
                    '心梗', '脑梗', '脑出血', '肿瘤', '癌症',
                    '隐血', '胎动'}
    
    def symptom_score(s):
        if s in HIGH_PRIORITY: return 3
        if s in MED_PRIORITY: return 2
        if s in LOW_PRIORITY: return 1
        return 1  # unknown
    
    # 按重要性排序
    scored = [(symptom_score(s), s) for s in real_symptoms]
    scored.sort(key=lambda x: (-x[0], len(x[1])))  # high score first, then shorter
    
    # 选1-2个，避免同类重复（如"疼痛"+"头痛"）
    selected = []
    selected_bases = set()  # track symptom "base" to avoid pairing related
    
    def symptom_base(s):
        for core in ['头痛', '胸痛', '腹痛', '腰痛', '关节痛', '咽痛', '耳痛', '眼痛', '牙痛']:
            if core in s: return 'site_pain'
        if '痛' in s: return 'pain'
        if '咳' in s or '痰' in s: return 'cough'
        if '热' in s or '烧' in s: return 'fever'
        if '吐' in s or '恶心' in s: return 'nausea'
        if '泻' in s or '便' in s or '秘' in s or '腹胀' in s: return 'gi'
        if '晕' in s: return 'dizzy'
        if '肿' in s or '水' in s: return 'edema'
        if '血' in s: return 'bleeding'
        if '呼吸' in s or '喘' in s or '气' in s: return 'resp'
        if '心' in s or '悸' in s or '胸' in s: return 'cardiac'
        return s[:2]
    
    for score, sym in scored:
        base = symptom_base(sym)
        if base not in selected_bases:
            selected.append(sym)
            selected_bases.add(base)
            # 仅当第一个症状评分较低(<3)时才补充第二个（HIGH优先级症状自足）
            if len(selected) >= 2:
                break
            if len(selected) == 1 and score >= 3:
                break  # 第一个症状已是HIGH优先级，足够完整
    
    if not selected:
        selected = [scored[0][1]] if scored else real_symptoms[:1]
    
    if not selected:
        selected = real_symptoms[:2]
    
    # 组装
    if len(selected) == 1:
        cc = f"{selected[0]}{time_str}" if time_str else selected[0]
    else:
        cc = f"{selected[0]}伴{selected[1]}{time_str}" if time_str else f"{selected[0]}伴{selected[1]}"
    
    # 15字截断
    if len(cc) > 15:
        # 尝试只用第一个症状
        cc = f"{selected[0]}{time_str}" if time_str else selected[0]
    
    return cc


def generate_fallback_chief_complaint(raw_text: str, info: object = None) -> str:
    """主诉规则兜底：从对话中提取症状+时间"""
    # 内联症状提取，避免循环导入
    symptom_keywords_local = {
        "咳嗽", "咳痰", "发热", "头痛", "头晕", "胸痛", "胸闷",
        "腹痛", "恶心", "呕吐", "腹泻", "便秘", "血尿", "水肿",
        "出血", "疼痛", "麻木", "乏力", "消瘦", "黄疸", "心悸",
        "气短", "气促", "喘息", "呼吸困难", "寒战", "皮疹",
        "瘙痒", "关节痛", "腰痛", "背痛", "昏迷", "意识",
        "黄痰", "白痰", "血痰", "干咳", "咽痒", "咽痛", "喉咙痛",
        "腹胀", "反酸", "嗳气", "食欲不振", "厌食",
    }
    # 口语化症状变体（含非标准子串，如"胸口痛"不含"胸痛"）
    colloquial_variants = {
        "胸口痛": "胸痛", "胸口疼": "胸痛",
        "肚子疼": "腹痛", "肚子痛": "腹痛",
        "脑袋疼": "头痛", "头胀": "头痛", "头疼": "头痛",
        "发烧": "发热", "烧": "发热",
        "喘不上气": "呼吸困难", "上不来气": "呼吸困难",
        "想吐": "恶心", "吐": "呕吐",
        "拉肚子": "腹泻", "拉稀": "腹泻",
        "没劲": "乏力", "累": "乏力",
        "心慌": "心悸", "心跳快": "心悸",
        "咳": "咳嗽", "冒汗": "出汗",
        "咳黄痰": "咳痰", "咳白痰": "咳痰", "有痰": "咳痰",
        "嗓子疼": "咽痛", "嗓子痛": "咽痛", "喉咙疼": "咽痛",
        "胃不舒服": "腹痛", "胃疼": "腹痛", "胃痛": "腹痛",
        "发冷": "寒战", "打哆嗦": "寒战",
    }
    symptoms = set()
    for s in symptom_keywords_local:
        if s in raw_text:
            symptoms.add(s)
    for variant, standard in colloquial_variants.items():
        if variant in raw_text:
            symptoms.add(standard)
    
    # 支持中文数字+阿拉伯数字的时长
    duration = ""
    # 先尝试阿拉伯数字模式
    m = re.search(r'(\d+)\s*(小时|分钟|[天周月年])', raw_text)
    if m:
        duration = m.group(1) + m.group(2)
    else:
        # 尝试中文数字模式：三个→3, 两→2, 几→''
        cn_num_map = {
            '一': '1', '二': '2', '两': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9',
            '十': '10', '半': '',
        }
        # 先尝试"X个星期"模式
        week_m = re.search(r'([一二两三四五六七八九十半])\s*(个)?\s*星期', raw_text)
        if week_m:
            cn = week_m.group(1)
            num = cn_num_map.get(cn, '')
            duration = f'{num}周' if num else '半周'
        else:
            for cn, num in cn_num_map.items():
                m2 = re.search(f'{cn}\s*(个)?\s*(小时|分钟|[天周月年])', raw_text)
                if m2:
                    unit = m2.group(2)
                    duration = f'{num}{unit}' if num else f'半{unit}'
                    break
    
    if symptoms and duration:
        symptom_list = list(symptoms)
        # 症状优先级排序：主要症状在前
        priority_symptoms = ["发热", "咳嗽", "咳痰", "胸痛", "腹痛", "头痛", "头晕", 
                            "恶心", "呕吐", "腹泻", "皮疹", "关节痛", "腰痛"]
        sorted_symptoms = sorted(symptom_list, 
                                key=lambda x: priority_symptoms.index(x) if x in priority_symptoms else 999)
        
        if len(sorted_symptoms) >= 2:
            # 如果有发热，发热放前面
            if "发热" in sorted_symptoms:
                sorted_symptoms.remove("发热")
                sorted_symptoms.insert(0, "发热")
            return f"{sorted_symptoms[0]}伴{sorted_symptoms[1]}{duration}"
        return f"{sorted_symptoms[0]}{duration}"
    
    # 只有症状没有时间
    if symptoms:
        symptom_list = list(symptoms)
        priority_symptoms = ["发热", "咳嗽", "咳痰", "胸痛", "腹痛", "头痛", "头晕",
                            "恶心", "呕吐", "腹泻", "皮疹", "关节痛", "腰痛"]
        sorted_symptoms = sorted(symptom_list,
                                key=lambda x: priority_symptoms.index(x) if x in priority_symptoms else 999)
        if len(sorted_symptoms) >= 2:
            return f"{sorted_symptoms[0]}伴{sorted_symptoms[1]}"
        return sorted_symptoms[0]
    
    # 无症状但有复查/随访场景 → 生成对应主诉
    if "乳腺癌术后" in raw_text:
        return "乳腺癌术后复查"
    if "复查" in raw_text or "复诊" in raw_text:
        disease_match = re.search(r'(高血压|糖尿病|乙肝|冠心病|慢阻肺|肾病|甲亢|肿瘤|乳腺癌)\s*(复查|复诊)', raw_text)
        if disease_match:
            return f"{disease_match.group(1)}{disease_match.group(2)}"
        # 乳腺癌术后X年复查
        bc_match = re.search(r'乳腺癌术后[\d年]+[，。]?\s*(?:今日)?\s*(复查|复诊)', raw_text)
        if bc_match:
            return f"乳腺癌术后{bc_match.group(1)}"
        if "乳腺癌术后" in raw_text:
            return "乳腺癌术后复查"
        # 高血压/糖尿病复查（病史多年+今日血压/血糖）
        if re.search(r'(高血压|糖尿病).{0,10}(?:年|史).{0,20}(?:血压|血糖)', raw_text):
            diseases = []
            if "高血压" in raw_text:
                diseases.append("高血压")
            if "糖尿病" in raw_text:
                diseases.append("糖尿病")
            if diseases:
                return "、".join(diseases) + "复查"
        # 通用复查模式
        if "复查" in raw_text:
            return "复查"
    if "随访" in raw_text:
        return "随访"
    if "携带者" in raw_text and ("ALT" in raw_text or "乙肝" in raw_text):
        if "复查" in raw_text:
            return "乙肝携带者复查"
        # 检测是否有异常值
        alt_match = re.search(r'ALT\s*(\d+)', raw_text)
        ast_match = re.search(r'AST\s*(\d+)', raw_text)
        has_abnormal = False
        if alt_match:
            try:
                if int(alt_match.group(1)) > 40:
                    has_abnormal = True
            except ValueError:
                pass
        if ast_match and not has_abnormal:
            try:
                if int(ast_match.group(1)) > 40:
                    has_abnormal = True
            except ValueError:
                pass
        if has_abnormal or re.search(r'(HBV|DNA)\s*\d{4,}', raw_text):
            return "乙肝携带者复查"
        return "乙肝携带者随访"
    
    return ""

    
    return "需结合临床表现及重复检测确认"

CHIEF_COMPLAINT_SYSTEM_PROMPT = """你是一名病历质控专家。请根据对话或叙述文本提取规范的主诉。

核心原则：
1. 主诉 = 患者本次就诊最主要的原因（症状/体征/复查目的）+ 持续时间
2. 必须转换为规范医学术语，禁止照抄口语原文
3. 仅提取本次就诊的主诉，慢性病/既往病史不出现在主诉中（除非是复查/随访目的）
4. 体检/复查/术后随访等非症状性就诊，保留"体检""复查""术后"关键词

格式要求：
- 格式：核心症状+时长，如"胸痛3小时"、"咳嗽1周"
- 最多15字。超过则只取最核心的1个症状+时长
- 如果输入是医患对话，提取患者主动诉说的核心不适
- 忽略医生提问、否认症状、既往病史、慢性病名称
- 伴随症状当与核心主诉关系紧密时保留（如常见的“腹泻伴呕吐”、“腹痛伴反酸”、“咳嗽伴喘息”等），避免过度裁剪导致信息丢失
- 模糊不适（如"浑身不舒服""吃饭不香"）→ 提取为"全身不适"等简洁表述

口语标准化表：
- "胸口疼/胸口痛/心口疼"→"胸痛"
- "肚子疼/胃疼/小腹痛"→"腹痛"
- "拉肚子/拉稀"→"腹泻"
- "发烧/烧到XX度"→"发热"
- "脑袋疼/头疼"→"头痛"
- "没劲/没精神"→"乏力"
- "冒汗/出汗/大汗"→"出汗"
- "耳朵嗡嗡响"→"耳鸣"
- "听不清/听不见"→"听力下降"
- "天旋地转/转圈晕"→"眩晕"
- "喘不上气/吸气费劲"→"呼吸困难"
- "尿不出来/拉不出尿"→"排尿困难"
- "咳了一个礼拜/咳了好几天"→"咳嗽"

对话示例：
输入：大夫，我胸口痛，大概有三小时了吧，一直冒汗
输出：胸痛3小时

输入：医生：怎么了？患者：拉肚子两天了，水一样的。医生：吐吗？患者：不吐。
输出：腹泻2天

输入：医生：哪里不舒服？患者：右小腿被车撞了一下，现在很疼，好像歪了。医生：多久了？患者：一小时前。
输出：右小腿外伤1小时

输入：医生：发烧吗？患者：38度5。医生：咳嗽吗？患者：咳了一周，有黄痰。
输出：咳嗽1周

输入：患者：医生，我头晕，天旋地转的，站起来就晕，躺下来好点，有三天了，还恶心，吐了一次。医生：有没有耳鸣？患者：左耳嗡嗡响。
输出：眩晕3天

输入：患者：大夫，我这几天浑身不舒服，不得劲，吃饭也不香，睡觉也不好。
输出：全身不适

输入：患者：医生，我肝癌术后半年了，这次来复查。
输出：肝癌术后复查

输入：患者：我乳腺癌切除一年了，来复查一下。医生：有没有不舒服？
输出：乳腺癌术后复查

输入：患者：我上周体检发现肺里有个结节，来看看。医生：多大？
输出：体检发现肺结节

输入：患者：大夫，我膝盖疼，两个膝盖都疼，上下楼梯的时候厉害，蹲下去站不起来，有半年了。
输出：双膝关节痛半年

输入：患者：大夫，我咳嗽不停，还喘不上气，有一个月了。
输出：咳嗽伴喘息1月余

输入：患者：大夫，我左上腹隐隐疼，还反酸水，有两周了。
输出：左上腹隐痛伴反酸2周

输入：患者：医生，我晚上睡觉打呼噜，有时候会憋醒，有两三年了。
输出：睡眠打鼾伴呼吸暂停2年

输入：患者：医生，我右眼模糊看不清，有一个礼拜了。医生：有糖尿病吗？患者：有，好几年了。
输出：视物模糊1周

错误示例：
输入：患者：医生，我右眼模糊看不清，有一个礼拜了。医生：有糖尿病吗？患者：有，好几年了。
输出：糖尿病伴高血压  ← 错误！主诉是视物模糊，不是慢性病
输出：胸口疼痛痛  ← 禁止重复字符
输出：胸痛伴出汗3小时  ← 伴随症状不放入主诉

只输出主诉文本，不要解释，不要引号。"""



class APIClient:
    """DeepSeek API 调用封装 - 混合架构的 API 语义层
    规则做提取+校验，API 做诊断推断+主诉生成
    """
    
    # 调用计数器（类级别，测试用）
    cc_call_count = 0
    diag_call_count = 0
    
    @classmethod
    def reset_counts(cls):
        cls.cc_call_count = 0
        cls.diag_call_count = 0
    
    @staticmethod
    def post_process_chief_complaint(raw: str, raw_text: str) -> str:
        """主诉后处理：修正常见错误"""
        if not raw:
            return ""
        # 去除诊断名称（如果混入）
        diagnosis_terms = ["肺炎", "心肌梗死", "脑出血", "脑梗死", "糖尿病", "高血压", "冠心病", "阑尾炎"]
        for term in diagnosis_terms:
            raw = raw.replace(term, "")
        # 去除"建议""考虑""可能"等医生用语
        raw = re.sub(r'建议.*', '', raw)
        raw = re.sub(r'考虑|可能', '', raw)
        # 确保有时间词
        if not re.search(r'\d+\s*[天周月年小时分钟]', raw):
            # 尝试从原文提取时间词
            time_match = re.search(r'(\d+\s*(?:天|周|个月|月|年|小时|分钟))', raw_text)
            if time_match:
                raw = raw.strip() + time_match.group(1)
        # 清理多余空格
        raw = re.sub(r'\s+', '', raw)
        if raw and len(raw) >= 4:
            return raw
        return ""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=get_api_key(),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        )
        self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    def generate_chief_complaint(self, extracted: 'ExtractedInfo', raw_text: str) -> str:
        """按需调 API 生成规范主诉"""
        cc = extracted.chief_complaint or ""
        
        # 口语化指示词
        colloquial_indicators = ['不得劲', '不舒服', '有点', '有些', '隐隐', '好几天', '浑身', '吃不下', '没劲儿', '感觉', '不香', '不好']
        has_colloquial = any(ind in cc for ind in colloquial_indicators)
        
        # 质量检查：是否含时长信息
        has_temporal = bool(re.search(r'\d+\s*(天|周|月|年|小时|分钟|秒)', cc))
        
        # 场景词：复查/术后/随访不需要时长
        scene_words = ['复查', '术后', '随访', '体检']
        is_scene = any(w in cc for w in scene_words)
        
        # 触发条件：
        # 1. 空或太短
        # 2. 含口语词
        # 3. 无时长且非场景
        # 4. 过长（>20字）
        # 5. 含"待查""未能识别"
        should_call_api = (
            not cc
            or len(cc) < 4
            or has_colloquial
            or (not has_temporal and not is_scene)
            or len(cc) > 20
            or '待查' in cc
            or '未能识别' in cc
            or '待补充' in cc
        )
        
        if not should_call_api:
            return ""
        
        APIClient.cc_call_count += 1
        
                # Build prompt inline
        is_dialogue = bool(re.search(r'(?:医生|患者|家属|护士|病人)(?!姓名)[^：:\n]{0,3}[：:：]', raw_text))
        if is_dialogue:
            prompt = f"""以下是一段医患对话，请从中提取患者的核心主诉。

对话内容：
{raw_text[:500]}

请给出规范主诉（核心症状+时长）："""
        else:
            symptoms = extracted.positive_symptoms or []
            symptoms_str = "、".join(symptoms[:8]) if symptoms else ""
            temporal = re.findall(r'(\d+\s*(?:天|周|个月|月|年|小时|分钟))', raw_text)
            temporal_str = "、".join(temporal) if temporal else "无明确时长"
            prompt = f"""患者信息：
- 症状：{symptoms_str or '未提取到明确症状'}
- 时长：{temporal_str}
- 原文：{raw_text[:200]}

请生成规范主诉："""
        if not prompt:
            return ""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": CHIEF_COMPLAINT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}],

                temperature=0.0,
                max_tokens=200
            )
            result = response.choices[0].message.content.strip()
            result = result.strip('"\'')
            # 安全套话检测：API输出套话时，触发规则兜底
            if is_safe_phrase(result):
                fallback = generate_fallback_chief_complaint(raw_text)
                if fallback:
                    return fallback
                return ""
                        # 过滤：不能比原主诉还差
            if result and len(result) >= 4 and result != cc:
                return result
            return ""
        except Exception as e:
            logger.error(f"API 主诉生成失败: {e}")
            return ""
    
    @staticmethod
    def extract_primary_diagnosis(api_output: str) -> str:
        """从API输出中提取第一诊断"""
        if not api_output:
            return ""
        # 匹配"第一诊断"后的内容
        m = re.search(r'第一诊断.*?[：:]\s*(.+?)(?:|$)', api_output)
        if m:
            diag = m.group(1).strip()
            return diag
        # 匹配"1."开头的第一行
        m = re.search(r'[1．1.]\s*(.+?)(?:|$)', api_output)
        if m:
            diag = m.group(1).strip()
            diag = re.sub(r'（.*?）|\(.*?\)', '', diag).strip()
            return diag
        # 兜底：取第一行非空内容
        lines = [l.strip() for l in api_output.split('') if l.strip()]
        if lines:
            diag = lines[0]
            diag = re.sub(r'^[1．1.]\s*', '', diag).strip()
            return diag
        return ""

    def infer_diagnosis(self, extracted, raw_text):
        """LLM诊断推断"""
        pass


    def infer_diagnosis(self, extracted: 'ExtractedInfo', raw_text: str) -> str:
        """结构化LLM诊断推断"""
        # Build prompt inline
        parts = []
        if extracted.gender and extracted.gender != "待补充":
            parts.append(f"患者{extracted.gender}")
        if extracted.age and extracted.age != "待补充":
            parts.append(f"{extracted.age}岁")
        pos = extracted.positive_symptoms or []
        neg = extracted.negative_symptoms or []
        if pos:
            parts.append(f"{'、'.join(pos)}")
        if neg:
            parts.append(f"无{'、'.join(neg)}")
        key_signs = []
        if '压痛' in raw_text: key_signs.append("压痛")
        if '反跳痛' in raw_text: key_signs.append("反跳痛阳性")
        if '颈抵抗' in raw_text: key_signs.append("颈抵抗阳性")
        if '昏迷' in raw_text: key_signs.append("昏迷")
        if '意识模糊' in raw_text: key_signs.append("意识模糊")
        if '啰音' in raw_text and '消失' not in raw_text and '未及' not in raw_text:
            key_signs.append("肺部啰音")
        if key_signs:
            parts.append(f"体征：{'、'.join(key_signs)}")
        bp = re.search(r'血压\D*(\d+)/(\d+)', raw_text)
        if bp:
            parts.append(f"血压{bp.group(1)}/{bp.group(2)}")
        hr = re.search(r'心率\D*(\d+)', raw_text)
        if hr:
            parts.append(f"心率{hr.group(1)}次/分")
        if extracted.blood_sugar:
            parts.append(f"空腹血糖{extracted.blood_sugar}")
        if extracted.past_history and extracted.past_history != "待补充":
            parts.append(f"既往：{extracted.past_history}")
        if '术后' in raw_text:
            parts.append("术后复查")
        narrative = "，".join(parts)
        raw_snippet = raw_text.strip()[:300]
        if len(raw_snippet) > 300:
            raw_snippet = raw_snippet[:300] + "..."
        prompt = f"""患者信息：{narrative}

原文：{raw_snippet}

请给出诊断结论："""
        
        if not prompt:
            return ""
        
        APIClient.diag_call_count += 1
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": """你是一名全科临床医生。请根据患者信息给出倾向性诊断。

基本规则：
1. 基于主诉症状和伴随体征，推断最可能的诊断，标注"可能"或"待排"
2. 推断诊断时使用大类（如"肺炎"而非"支气管肺炎"），但患者自述或外院已确诊的具体诊断必须原文引用
3. 患者自述或外院已确诊的疾病，直接引用并标注"（来源：患者自述）"
4. 信息不足或症状模糊（如仅有“全身不适”“乏力”等非特异性症状）时，输出“不适待查，建议进一步检查”而非强行给出具体诊断
5. 外伤/创伤场景：优先列出具体损伤，而非仅给并发症
6. 只输出诊断文本，不要解释"""},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            result = response.choices[0].message.content.strip()
            # Post-process: fix common LLM output issues
            if "待补充可能" in result:
                result = result.replace("待补可能", "不适待查")
            if "不适待查可能" in result:
                result = result.replace("不适待查可能", "不适待查")
            if result == "待补充":
                result = "不适待查，建议进一步检查"
            return result

        except Exception as e:
            logger.warning(f"LLM诊断失败: {e}")
            return ""

def chinese_num_to_arabic(cn_str):
    cn_num_map = {"零":0,"一":1,"二":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"两":2}
    cn_unit_map = {"十":10,"百":100,"千":1000,"万":10000,"亿":100000000}
    if not cn_str:
        return 0
    if cn_str.isdigit():
        return int(cn_str)
    result = 0
    current_num = 0
    section_num = 0
    for ch in cn_str:
        if ch in cn_num_map:
            current_num = cn_num_map[ch]
        elif ch in cn_unit_map:
            unit = cn_unit_map[ch]
            if current_num == 0:
                current_num = 1
            if unit >= 10000:
                section_num = (section_num + current_num) * unit
                current_num = 0
                if unit >= 100000000:
                    result += section_num
                    section_num = 0
            else:
                section_num += current_num * unit
                current_num = 0
    result += section_num + current_num
    return result


class InfoExtractor:
    """信息提取器 - 采用正则+LLM双重提取策略"""
    
    # 口语化主诉 → 标准化主诉映射表
    COLLOQUIAL_TO_STANDARD = {
        '浑身不得劲': '全身不适',
        '浑身不舒服': '全身不适',
        '不舒服': '不适',
        '隐隐作痛': '隐痛',
        '隐隐疼': '隐痛',
        '痛': '疼痛',
        '好几天': '数日',
        '好几天了': '数日',
        '有些': '',
        '有点': '',
        '感觉': '',
        '觉得': '',
        '疼得厉害': '疼痛剧烈',
        '疼': '疼痛',
        '拉肚子': '腹泻',
        '吃不下饭': '食欲不振',
        '吃不下': '食欲不振',
        '没劲儿': '乏力',
        '没劲': '乏力',
        '喘不上气': '呼吸困难',
        '喘': '气促',
        '心慌': '心悸',
        '身上痒': '皮肤瘙痒',
        '起皮疹': '皮疹',
        '起疹子': '皮疹',
    }
    


    @classmethod
    def standardize_chief_complaint(cls, text: str) -> str:
        """将口语化主诉标准化"""
        if not text:
            return text
        
        result = text
        # 按长度降序排序，优先匹配长的短语
        for slang, standard in sorted(cls.COLLOQUIAL_TO_STANDARD.items(), key=lambda x: -len(x[0])):
            result = result.replace(slang, standard)
        
        # 清理多余空格和逗号
        result = re.sub(r'\s+', '', result)
        result = re.sub(r'[，,]{2,}', '，', result)
        result = result.strip('，。！？ ')
        
        return result or '不适'
    
    # 正则模式库
    PATTERNS = {
        "gender": [
            r'性别[\s：:]?(男|女)',
            r'(男|女)(?:性)?',
            r'(?:患者|患儿|病人|其子|其女|夫妻|妻子|丈夫)[\s\S]{0,5}?([男女])',
            r'^([男女])',
            r'([男女])\s*[，,、]',],

        "age": [
            r'年龄[\s：:]?(\d{1,3})',
            r'(\d{1,3})岁\s*[男女]',
            r'(\d{1,3})岁',
            r'(\d{1,3})周岁',
            r'[男女][\s，,、]*(\d{1,3})\s*岁',
            r'(\d{1,3})\s*岁(?:余|多|许|左右|以上|以下)?',],

        "chief_complaint": [
            r'(头痛|胸痛|腹痛|发热|咳嗽|头晕|乏力|恶心|呕吐|呼吸困难|心悸|腹泻|便秘|关节痛|腰痛|视力模糊|听力下降|鼻出血|牙龈出血|皮肤瘙痒|水肿|麻木|抽搐|昏迷|意识模糊|食欲不振|体重下降|口干|多饮|多尿|失眠|记忆减退|情绪低落|胸闷|气短|腹胀|黄疸|皮疹|寒战|胎动减少|胎动异常|车祸伤后|外伤后|摔倒后|跌倒后)(\d+天|\d+周|\d+月|\d+小时|\d+年|\d+分钟|\d+秒)',
            r'主诉[\s：:]?([^\d]+?)(\d+天|\d+周|\d+月|\d+小时|\d+年)',
            r'(?:因)?["\']([^\d]+?)(\d+天|\d+周|\d+月|\d+小时|\d+年)["\']',],

        "blood_sugar": [
            r'血糖[\s：:]?(\d+\.?\d*)\s*(mmol/L|mg/dL|mmol|L)?',
            r'血糖空腹[\s：:]?(\d+\.?\d*)',
            r'空腹血糖[\s：:]?(\d+\.?\d*)',
            r'空腹[\s：:]?(\d+\.?\d*)\s*(mmol/L|mg/dL)?',
            r'FBG[\s：:]?(\d+\.?\d*)'],

        "blood_sugar_2h": [
            r'餐后[\s：:]?(\d+小时|\d+h)?[\s：]?血糖[\s：:]?(\d+\.?\d*)',
            r'餐后2小时血糖[\s：:]?(\d+\.?\d*)',
            r'餐后[\s：:]?(\d+小时|\d+h)?[\s：]?(\d+\.?\d*)',
            r'饭后[\s：:]?(\d+小时|\d+h)?[\s：]?血糖[\s：:]?(\d+\.?\d*)',
            r'饭后2小时血糖[\s：:]?(\d+\.?\d*)',
            r'饭后[\s：:]?(\d+小时|\d+h)?[\s：]?(\d+\.?\d*)',
            r'饭后\s*两\s*小时[\s：:]?(\d+\.?\d*)',
            r'餐后\s*\d*\s*h\s*[\s：:]?(\d+\.?\d*)',
            r'2hPG[\s：:]?(\d+\.?\d*)'],

        "hba1c": [
            r'糖化血红蛋白[\s：:]?(\d+\.?\d*)\s*%?',
            r'HbA1c[\s：:]?(\d+\.?\d*)\s*%?',
            r'糖化[\s：:]?(\d+\.?\d*)\s*%?'],

        "blood_pressure": [
            r'血压[\s：:]?(\d{2,3})/(\d{2,3})\s*(mmHg)?',
            r'(\d{2,3})/(\d{2,3})\s*(mmHg)?',
            r'收缩压[\s：:]?(\d{2,3})[\s：]?舒张压[\s：:]?(\d{2,3})'],

        "heart_rate": [
            r'心率[\s：:]?(\d{1,3})',
            r'脉搏[\s：:]?(\d{1,3})',
            r'P[\s：:]?(\d{1,3})'],

        "respiratory_rate": [
            r'呼吸[\s：:]?(\d{1,3})',
            r'R[\s：:]?(\d{1,3})'],

        "temperature": [
            r'体温[\s：:]?(\d+\.?\d*)',
            r'(?<![A-Za-z])T[\s：:]?(\d+\.?\d*)',
            r'发热(\d+\.?\d*)度'],

        "symptoms": [
            # 基础症状
            r'(头痛|胸痛|腹痛|腹胀|发热|咳嗽|头晕|乏力|恶心|呕吐|呼吸困难|心悸|腹泻|便秘|关节痛|腰痛|视力模糊|听力下降|鼻出血|牙龈出血|皮肤瘙痒|水肿|麻木|抽搐|昏迷|意识模糊|食欲不振|体重下降|口干|口渴|多饮|多尿|失眠|记忆减退|情绪低落|胸闷|气短|腹胀|黄疸|皮疹|寒战|出汗|面色苍白|出冷汗|胎动减少|胎动|隐痛|隐血|刺痛|胀痛|绞痛|灼痛|抽痛|坠痛|反酸|烧心|打嗝|嗳气|尿频|尿急|尿痛|血尿)',
            # 症状修饰短语
            r'(咳黄痰|咳白痰|咳血痰|痰中带血|干咳|喘息|胸闷气短|呼吸困难|心慌心悸|隐隐作痛)',
            # 发热程度
            r'(发热最高(\d+\.?\d*)度|体温(\d+\.?\d*)度|T(\d+\.?\d*)度)',
            # 外伤相关症状
            r'(疼痛|肿胀|活动受限|摔伤|外伤|骨折|扭伤)'],

        "medical_history": [
            r'(既往有|曾患|以前有)([\u4e00-\u9fa5]+?)病(史)?(\d+年)?',
            r'(高血压|糖尿病|冠心病|心脏病|脑梗死|脑出血|慢性支气管炎|肺气肿|哮喘|肝炎|肾炎)(病史)?(\d+年)?',
            r'(高血压|糖尿病|冠心病)史(\d+年)?',
            r'(既往[\u4e00-\u9fa5]+?病史)',
            r'(有[\u4e00-\u9fa5]+?病史(\d+年)?)',
            r'有(高血压|糖尿病|冠心病|心脏病|支气管炎|哮喘|肝炎|肾炎|肾病|脑梗死|脑出血)([，。；]|$)'],

        "imaging_exam": [
            r'(胸片|胸部CT|头颅CT|头颅MRI|腹部B超|心脏彩超|超声)示([^\，。；]+?)(阴影|病灶|异常|改变)?',
            r'(检查示|提示|可见)([\u4e00-\u9fa5]+?阴影|[\u4e00-\u9fa5]+?病灶|[\u4e00-\u9fa5]+?异常|[\u4e00-\u9fa5]+?改变)',
            r'(胸片示|胸部CT示|头颅CT示)([\u4e00-\u9fa5]+?阴影)'],

        "fever_temp": [
            r'(发热|体温)最高(\d+\.?\d*)[度℃°]?',
            r'最高(\d+\.?\d*)[度℃°]?',
            r'(?<![A-Za-z])(T|体温)(\d+\.?\d*)°?C?',
            r'(\d+\.?\d*)[℃°]',],

        "diagnosis_source": [
            r'(诊断为|考虑为|疑似为|诊为)([^，。；]+(?:[，、][^，。；]+)*)',
            r'(外院诊断|当地医院诊断|医院诊断)([^，。；]+(?:[，、][^，。；]+)*)',
            r'(诊断)([^，。；]*糖尿病)',
            r'(诊断为|考虑为|疑似为|确诊为)([^，。；]+(?:[，、][^，。；]+)*)',],

        "surgery_history": [
            r'(行[\u4e00-\u9fa5]+术)(?:后)?',
            r'(?:有|曾)([\u4e00-\u9fa5]+术)(?:史)?',
            r'([\u4e00-\u9fa5]+术后)',],

        "treatment_plan": [
            r'(给予|给予治疗|治疗方案|治疗计划)([\u4e00-\u9fa5，。；]+?)(？|。|；|$)',
            r'(服用|口服|静滴|肌注)([\u4e00-\u9fa5]+?)(片|mg|g|ml)?'],

        "patient_name": [
            r'患者姓名[：:]?\s*([\u4e00-\u9fa5]{2,4})',
            r'姓名[：:]?\s*([\u4e00-\u9fa5]{2,4})',
            r'患者([\u4e00-\u9fa5]{2,4})',
            r'(?:我|他|她)(?:叫|是)([\u4e00-\u9fa5]{2,4})',
            r'(?:叫|是)([\u4e00-\u9fa5]{2,4})(?=[，,。；;]|$)'],

        "phone": [
            r'(?:电话|手机|联系电话)[：:]?\s*(1[3-9]\d)[\- ]?(\d{4})[\- ]?(\d{4})',
            r'(1[3-9]\d)[\- ]?(\d{4})[\- ]?(\d{4})',
            r'(0\d{2,3})[\- ]?(\d{7,8})',],

        "id_card": [
            r'身份证[：:是号码](\d{17}[\dXx])',
            r'身份证号[：:是号码](\d{17}[\dXx])',
            r'(\d{17}[\dXx])(?!\d)'],

        # 新增检验指标提取
        "blood_oxygen": [
            r'血氧[\s：:]?(\d+)%?',
            r'SpO2[\s：:]?(\d+)%?',],

        "tsh": [
            r'TSH[\s：:]?([\d.]+)',],

        "ft3": [
            r'FT3[\s：:]?([\d.]+)',],

        "ft4": [
            r'FT4[\s：:]?([\d.]+)',],

        "cea": [
            r'CEA[\s：:]?([\d.]+)',],

        "ca199": [
            r'CA199[\s：:]?(\d+)',],

        "creatinine": [
            r'(?:肌酐|Cr)[\s：:]?([\d.]+)',],

        "bun": [
            r'(?:尿素氮|BUN)[\s：:]?([\d.]+)',],

        "afp": [
            r'AFP[\s：:]?([\d.]+)',],

        "pt": [
            r'PT[\s：:]?([\d.]+)\s*s',],

        "aptt": [
            r'APTT[\s：:]?([\d.]+)\s*s',],

        "fib": [
            r'FIB[\s：:]?([\d.]+)',],

        "d_dimer": [
            r'D[-]?二聚体[\s：:]?([\d.]+)',],

        "esr": [
            r'血沉[\s：:]?(\d+)',],

        "crp": [
            r'CRP[\s：:]?([\d.]+)',],

        "alt": [
            r'ALT[\s：:]?([\d.]+)',],

        "ast": [
            r'AST[\s：:]?([\d.]+)',],

        "tc": [
            r'TC[\s：:]?([\d.]+)',],

        "tg": [
            r'TG[\s：:]?([\d.]+)',],

        "ldl_c": [
            r'LDL[-]?C[\s：:]?([\d.]+)',],

        "hdl_c": [
            r'HDL[-]?C[\s：:]?([\d.]+)',],

        "uric_acid": [
            r'(?:尿酸|UA)[\s：:]?([\d.]+)',],

        "ca153": [
            r'CA153[\s：:]?([\d.]+)',],

        "hbv_dna": [
            r'HBV[-]?DNA[\s：:]?[<≤]?([\d.]+(?:\^?\d+)?)',],

        "wbc": [
            r'WBC[\s：:]?([\d.]+)\s*[×xX*]\s*10\^?9\s*(?:/\s*L)?',
            r'白细胞[\s：:]?([\d.]+)\s*[×xX*]\s*10\^?9\s*(?:/\s*L)?',
            r'WBC[\s：:]?([\d.]+)',
            r'白细胞[\s：:]?([\d.]+)',
            r'白细胞[高升]?[\s,，]+(\d+(?:\.\d+)?)',],

        "neutrophil_pct": [
            r'中性[\s：:]?(\d+)%',
            r'中性粒[\s：:]?(\d+)%',
            r'中性[粒高升]?[\s,，]+(\d+)%',],

        "amylase": [
            r'淀粉酶[\s：:]?(\d+)',
            r'血淀粉酶[\s：:]?(\d+)',],

        "bnp_value": [
            r'BNP[\s：:]?(\d+)',]

    }
    
    @classmethod
    def _preprocess_chinese_numbers(cls, text: str) -> str:
        """预处理：将BNP等指标后面的中文数字转换为阿拉伯数字"""
        cn_map = {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                  '六': '6', '七': '7', '八': '8', '九': '9', '零': '0', '两': '2'}
        cn_pattern = r'([一二三四五六七八九零十百千万两]+)'
        markers = ['BNP', 'NT-proBNP', 'proBNP']
        for marker in markers:
            m = re.search(re.escape(marker) + r'\s*' + cn_pattern, text)
            if m:
                cn_str = m.group(1)
                arabic = chinese_num_to_arabic(cn_str)
                text = text.replace(m.group(0), f"{marker} {arabic}")
        return text
    
    @classmethod
    def extract_symptoms_safe(cls, text: str) -> tuple:
        """带否定词的智能症状提取（增强版）
        
        处理逻辑：
        1. 按中文标点分句
        2. 每句独立判断否定词作用域
        3. 同一症状在多个分句中分别按否定/阳性处理
        4. 去重：同一症状在否定和阳性分句中都出现 → 取后者（最近语境）
        5. 额外捕获"无发热、咳嗽"这类省略否定词的并列结构
        6. 口语映射："吐"→呕吐，"拉"→腹泻，"烧"→发热
        
        Returns:
            (positive_list, negative_list)
        """
        # 所有症状关键词（保持与原有列表一致）
        symptom_words = ['头痛', '胸痛', '腹痛', '发热', '咳嗽', '头晕', '乏力', '恶心',
                        '呕吐', '呼吸困难', '心悸', '腹泻', '便秘', '关节痛', '腰痛',
                        '视力模糊', '听力下降', '鼻出血', '牙龈出血', '皮肤瘙痒', '水肿',
                        '麻木', '抽搐', '昏迷', '意识模糊', '食欲不振', '体重下降', '口干',
                        '口渴', '多饮', '多尿', '失眠', '记忆减退', '情绪低落', '胸闷',
                        '气短', '腹胀', '黄疸', '皮疹', '寒战', '出汗', '面色苍白', '出冷汗',
                        '疼痛', '肿胀', '活动受限', '摔伤', '外伤', '骨折', '扭伤',
                        '咳黄痰', '咳白痰', '咳血痰', '痰中带血', '干咳', '喘息',
                        '胸闷气短', '呼吸困难', '心慌心悸', '喘', '有痰', '精神差',
                        '吃奶少', '走路没劲', '没劲', '无力',
                        # 外伤/急诊
                        '伤口', '出血', '畸形', '车祸', '撞伤', '擦伤', '撕裂', '流血',
                        # 意识状态
                        '呼之可睁眼', '对答不清', '意识不清', '嗜睡', '昏睡', '谵妄',
                        # 神经系统
                        '偏瘫', '截瘫', '口角歪斜', '言语不清', '肢体无力',
                        # 疾病名称（用于否认疾病模式）
                        '高血压', '糖尿病', '冠心病', '肝炎', '乙肝', '脂肪肝',
                        '阑尾炎', '胰腺炎', '胃炎', '肺炎', '支气管炎',
                        '心梗', '脑梗', '脑出血', '肿瘤', '癌症',
                        # 口语映射目标词
                        '恶心', '呕吐', '腹泻', '发热',
                        # 严重程度修饰
                        '剧烈头痛', '压榨性胸痛',
                        '胎动减少', '胎动',
                        '隐痛', '隐血', '刺痛', '胀痛', '绞痛', '灼痛', '抽痛', '坠痛',
                        '反酸', '烧心', '打嗝', '嗳气',
                        '尿频', '尿急', '尿痛', '血尿',
                        # 部位+症状复合词
                        '咽痛', '耳痛', '眼痛', '牙痛', '颈痛', '肩痛', '膝痛', '踝痛', '腕痛', '肘痛', '背痛',
                        # 五官/呼吸道
                        '鼻塞', '流涕', '喷嚏', '耳鸣', '耳闷',
                        # 出血类
                        '便血', '黑便', '咯血',
                        # 腹部
                        '腹部不适', '下腹痛', '上腹痛', '右下腹痛', '反跳痛',
                        # 泌尿/生殖
                        '肉眼血尿', '阴道流血', '停经', '乳房胀痛',
                        # 消化道
                        '吞咽困难', '停止排气排便', '水样便',
                        # 其他常见
                        '声音嘶哑', '双下肢水肿', '下肢水肿', '视物模糊', '夜间盗汗', '口干眼干', '活动后气促', '活动后胸闷',
                        '眩晕', '膝关节痛', '膝痛', '活动后加重',]
        
        # 口语→标准映射
        SLANG_MAP = {'吐': '呕吐', '拉': '腹泻', '烧': '发热', '疼': '疼痛', '痛': '疼痛'}
        
        # 否定词集合
        NEGATION_WORDS = {'无', '没有', '否认', '未', '未见', '未曾', '并无', '不像', '不是'}
        
        # 文本归一化：口语/同义字→标准形式（在症状匹配前执行）
        TEXT_NORMALIZE = {
            # 疼→痛（部位+症状复合词）
            '头疼': '头痛', '胸疼': '胸痛', '肚子疼': '腹痛', '腰疼': '腰痛',
            '关节疼': '关节痛', '牙疼': '牙痛', '嗓子疼': '咽痛', '喉咙疼': '咽痛',
            '背疼': '背痛', '颈疼': '颈痛', '肩疼': '肩痛', '膝疼': '膝痛',
            '胃疼': '上腹痛', '腿疼': '下肢疼痛', '腹部疼': '腹痛',
            '耳疼': '耳痛', '眼疼': '眼痛', '肘疼': '肘痛', '腕疼': '腕痛', '踝疼': '踝痛',
            # 口语→标准
            '发烧': '发热', '发高烧': '高热',
            '拉肚子': '腹泻', '拉稀': '腹泻',
            '心慌': '心悸', '心跳快': '心悸',
            '喘不上气': '呼吸困难', '上不来气': '呼吸困难',
            '想吐': '恶心',
            '没劲': '乏力', '没劲儿': '乏力',
            '冒汗': '出汗',
            # 复合部位+症状
            '胸口痛': '胸痛', '胸口疼': '胸痛', '胸前痛': '胸痛',
            '前胸疼': '胸痛', '前胸痛': '胸痛',
            '后背疼': '背痛', '后背痛': '背痛',
            '后腰疼': '腰痛', '后腰痛': '腰痛',
            '脑袋疼': '头痛', '脑袋痛': '头痛',
            '全身疼': '全身疼痛', '浑身疼': '全身疼痛',
            # 其他口语
            '不得劲': '不适', '不舒服': '不适',
            '没食欲': '食欲不振',
            '睡不着': '失眠',
            '肿了': '肿胀', '有点肿': '肿胀',
        '脚肿': '双下肢水肿', '腿肿': '下肢水肿', '足肿': '下肢水肿',
        '脸肿': '面部水肿', '眼皮肿': '眼睑水肿', '眼肿': '眼睑水肿',
        '手肿': '手部水肿', '全身肿': '全身水肿',
        '腰酸': '腰痛', '腰有点酸': '腰痛', '背酸': '背痛',
        '茶色尿': '血尿', '小便泡沫': '泡沫尿', '尿泡沫': '泡沫尿',
        '咳了': '咳嗽', '烧到': '发热', '烧了': '发热',
        '膝盖疼': '膝关节痛', '膝盖痛': '膝关节痛',
        '蹲下去站不起来': '活动受限', '上下楼梯': '活动后加重',
        '天旋地转': '眩晕', '转圈': '眩晕',
        '模糊看不清': '视物模糊', '看不清': '视物模糊', '看不清东西': '视物模糊',
        '眼睛模糊': '视物模糊', '突然看不清': '视物模糊',

        '嗡嗡响': '耳鸣', '听不清': '听力下降',
        '吐了一次': '呕吐', '吐了': '呕吐',


            '心口疼': '胸痛', '心口痛': '胸痛',
        }
        # 按长度降序替换（避免"心慌"替换后"心悸"不被处理）
        for oral, std in sorted(TEXT_NORMALIZE.items(), key=lambda x: -len(x[0])):
            text = text.replace(oral, std)
        
        # 步骤1：分句
        sentences = re.split(r'[，,。；;、]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 步骤2：逐句判断极性
        # 用有序dict保留顺序，后面遇到同症状会覆盖之前的极性
        results = {}  # {symptom_name: is_negative}
        
        for sent in sentences:
            # 判断当前句子是否在否定词作用域内
            has_negation = any(
                sent.startswith(neg) or f'，{neg}' in f'，{sent}' or f'、{neg}' in f'、{sent}'
                for neg in NEGATION_WORDS
            )
            # "不"单独判断：排除"不适""不良"等复合词
            if not has_negation and '不' in sent:
                stripped = sent
                for compound in ['不适', '不良', '不行', '不好', '不足', '不规律']:
                    stripped = stripped.replace(compound, '')
                if stripped.startswith('不') or re.search(r'[，,、]不', stripped):
                    has_negation = True

            # 清理否定词前缀用于匹配症状
            clean_sent = sent
            for neg in NEGATION_WORDS:
                if clean_sent.startswith(neg):
                    clean_sent = clean_sent[len(neg):].lstrip('，、 ')
                clean_sent = re.sub(r'[，、]' + re.escape(neg), '', clean_sent)
            
            # 在当前分句中匹配症状
            found_symptoms = [s for s in symptom_words if s in clean_sent]
            
            # 口语映射：匹配"吐""拉""烧"等单字
            # 智能判断：仅当前缀+slang的2字复合词在symptom_words中时才保留
            for slang, standard in SLANG_MAP.items():
                if slang in clean_sent:
                    positions = [m.start() for m in re.finditer(re.escape(slang), clean_sent)]
                    is_compound = False
                    for pos in positions:
                        if pos > 0 and re.match(r'[一-鿿]', clean_sent[pos-1]):
                            compound = clean_sent[pos-1:pos+1]
                            # 仅当复合词是已知症状词时才保留（防止"节疼""盖疼"等无意义组合）
                            if compound in symptom_words and compound not in found_symptoms:
                                found_symptoms.append(compound)
                                is_compound = True
                    if is_compound:
                        continue
                    # 原逻辑：避免已匹配症状中的slang被重复映射
                    is_part_of_existing = any(slang in s for s in found_symptoms if len(s) > 1)
                    if not is_part_of_existing and standard not in found_symptoms:
                        found_symptoms.append(standard)
            
            for symptom in found_symptoms:
                results[symptom] = has_negation
        
        # 步骤2b：额外处理"不XX"模式中的口语词
        for slang, standard in SLANG_MAP.items():
            if re.search(rf'(?:不|没有|无|否认){slang}', text):
                results[standard] = True
        
        # 步骤3：额外处理"无发热、咳嗽"这类省略否定词的并列结构
        neg_scope = cls.extract_negation_scope(text)
        for symptom in neg_scope:
            results[symptom] = True  # 强制设为否定
        
        # 步骤4：按极性分类
        positive = [s for s, is_neg in results.items() if not is_neg]
        negative = [s for s, is_neg in results.items() if is_neg]
        
        return list(dict.fromkeys(positive)), list(dict.fromkeys(negative))
    
    @classmethod
    def extract_negation_scope(cls, text: str) -> list:
        """提取否定词作用域内的症状（处理'无发热、咳嗽'模式）
        
        例如："无发热、咳嗽" → ["发热", "咳嗽"]
        例如："否认恶心、呕吐" → ["恶心", "呕吐"]
        """
        NEGATION_WORDS = {'无', '没有', '否认', '未', '未见', '不', '未曾', '并无'}
        symptom_words = ['头痛', '胸痛', '腹痛', '发热', '咳嗽', '头晕', '乏力', '恶心',
                        '呕吐', '呼吸困难', '心悸', '腹泻', '便秘', '关节痛', '腰痛',
                        '皮疹', '抽搐', '意识模糊', '食欲不振', '腹胀', '黄疸', '寒战',
                        '出血', '水肿', '麻木', '胸闷', '气短', '咳痰', '有痰',
                        # 疾病名称
                        '高血压', '糖尿病', '冠心病', '肝炎', '乙肝', '脂肪肝',
                        '阑尾炎', '胰腺炎', '胃炎', '肺炎', '支气管炎',
                        '心梗', '脑梗', '脑出血', '肿瘤', '癌症',
                        # 严重程度（用于"不剧烈""不严重"模式）
                        '剧烈', '严重', '频繁', '明显', '显著', '持续', '影响活动',
                        # 口语映射
                        '呕吐', '腹泻', '发热', '疼痛',]
        SLANG_MAP = {'吐': '呕吐', '拉': '腹泻', '烧': '发热', '疼': '疼痛', '痛': '疼痛'}
        
        negated_symptoms = []
        # 匹配否定词后面的内容直到句末
        pattern = r'(?:' + '|'.join(re.escape(w) for w in NEGATION_WORDS) + r')([^，,。；;]*)'
        matches = re.findall(pattern, text)
        
        for phrase in matches:
            phrase = phrase.strip()
            # 在否定作用域内匹配症状
            for sym in symptom_words:
                if sym in phrase:
                    negated_symptoms.append(sym)
                    phrase = phrase.replace(sym, '')
            # 口语映射
            for slang, standard in SLANG_MAP.items():
                if slang in phrase and standard not in negated_symptoms:
                    negated_symptoms.append(standard)
        
        return negated_symptoms
    
    @classmethod
    def postprocess_symptoms_from_llm(cls, llm_text: str, original_input: str) -> str:
        """利用原始输入的否定信息修正LLM输出的症状文本
        
        Args:
            llm_text: LLM生成的症状描述文本
            original_input: 用户原始输入
            
        Returns:
            修正后的文本
        """
        # 从原始输入提取正确的极性
        positive, negative = cls.extract_symptoms_safe(original_input)
        correct_negatives = set(negative)
        
        # 对每个在原始输入中为阴性的症状，从LLM输出中移除
        for symptom in correct_negatives:
            # 替换"有XX"或"伴XX"为"无XX"
            llm_text = re.sub(f'有{symptom}', f'无{symptom}', llm_text)
            llm_text = re.sub(f'伴{symptom}，', '', llm_text)
            llm_text = re.sub(f'伴{symptom}', '', llm_text)
        
        # 清理多余标点
        llm_text = re.sub(r'，{2,}', '，', llm_text)
        llm_text = re.sub(r'，。', '。', llm_text)
        llm_text = llm_text.strip('， ')
        
        return llm_text
    
    @classmethod
    def extract_struct_exams(cls, text: str) -> tuple:
        """提取结构化的查体和检验数据"""
        physical = ""
        lab_parts = []
        
        # 查体引导词模式
        exam_patterns = [
            (r'查体[：:](.+?)(?=辅助检查|血常规|生化|胸片|CT|MRI|超声|B超|$)', 'physical'),
            (r'体格检查[：:](.+?)(?=辅助检查|血常规|生化|$)','physical'),
            (r'血常规[：:](.+?)(?=生化|胸片|CT|查体|$)', 'lab'),
            (r'生化[^：:]*[：:](.+?)(?=血常规|胸片|CT|乙肝|$)', 'lab'),
            (r'乙肝两对半[：:](.+?)(?=生化|血常规|$)', 'lab'),
            (r'血脂[：:]*(.+?)(?=生化|血常规|$)', 'lab'),
            (r'肾功[^：:]*[：:](.+?)(?=血脂|生化|$)', 'lab'),
            (r'肝功[^：:]*[：:](.+?)(?=肾功|血脂|生化|$)', 'lab'),
            (r'凝血[功能]*[：:](.+?)(?=生化|血常规|$)', 'lab'),
            (r'肿瘤标志物[：:](.+?)(?=生化|血常规|$)', 'lab'),
            (r'甲状腺功能[：:](.+?)(?=生化|血常规|甲状腺超声|$)', 'lab'),
            (r'甲功[：:](.+?)(?=生化|血常规|甲状腺超声|$)', 'lab'),
            (r'血气分析[：:](.+?)(?=生化|血常规|$)', 'lab'),
            (r'BNP[：:]*(.+?)(?=生化|血常规|$)', 'lab'),]

        
        for pattern, field in exam_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                content = match.group(1).strip()
                if field == 'physical':
                    physical = content
                elif field == 'lab':
                    lab_parts.append(content)
        
        lab = "；".join(lab_parts) if lab_parts else ""
        
        # 额外提取结构化检验数值对
        structured_pairs = cls.extract_lab_values(text)
        if structured_pairs:
            if lab:
                lab += "；" + structured_pairs
            else:
                lab = structured_pairs
        
        return physical, lab
    
    @classmethod
    def extract_lab_values(cls, text: str) -> str:
        """提取所有结构化检验数值对（名称+数值+单位）"""
        pairs = []
        
        lab_patterns = [
            # 血脂
            (r'(?:TC|总胆固醇)\s*[:：]?\s*([\d.]+)', '总胆固醇'),
            (r'(?:TG|甘油三酯)\s*[:：]?\s*([\d.]+)', '甘油三酯'),
            (r'(?:LDL-C?|低密度脂蛋白)\s*[:：]?\s*([\d.]+)', '低密度脂蛋白'),
            (r'(?:HDL-C?|高密度脂蛋白)\s*[:：]?\s*([\d.]+)', '高密度脂蛋白'),
            # 肾功能
            (r'(?:Cr|肌酐)\s*[:：]?\s*([\d.]+)', '肌酐'),
            (r'(?:BUN|尿素氮)\s*[:：]?\s*([\d.]+)', '尿素氮'),
            (r'(?:UA|尿酸)\s*[:：]?\s*([\d.]+)', '尿酸'),
            # 甲状腺功能
            (r'(?:TSH)\s*[:：]?\s*([\d.]+)', 'TSH'),
            (r'(?:FT3)\s*[:：]?\s*([\d.]+)', 'FT3'),
            (r'(?:FT4)\s*[:：]?\s*([\d.]+)', 'FT4'),
            # 肿瘤标志物
            (r'(?:AFP|甲胎蛋白)\s*[:：]?\s*([\d.]+)', 'AFP'),
            (r'(?:CEA|癌胚抗原)\s*[:：]?\s*([\d.]+)', 'CEA'),
            (r'(?:CA199|CA19-9)\s*[:：]?\s*([\d.]+)', 'CA199'),
            (r'(?:CA153|CA15-3)\s*[:：]?\s*([\d.]+)', 'CA153'),
            (r'(?:CA125|CA12-5)\s*[:：]?\s*([\d.]+)', 'CA125'),
            (r'(?:CA724|CA72-4)\s*[:：]?\s*([\d.]+)', 'CA724'),
            # 凝血功能
            (r'(?:PT)\s*[:：]?\s*([\d.]+)', 'PT'),
            (r'(?:APTT)\s*[:：]?\s*([\d.]+)', 'APTT'),
            (r'(?:FIB|纤维蛋白原)\s*[:：]?\s*([\d.]+)', 'FIB'),
            (r'(?:D-二聚体|D二聚体)\s*[:：]?\s*([\d.]+)', 'D-二聚体'),
            # 血气分析
            (r'(?:pH)\s*[:：]?\s*([\d.]+)', 'pH'),
            (r'(?:PaCO2|PCO2)\s*[:：]?\s*([\d.]+)', 'PaCO2'),
            (r'(?:PaO2|PO2)\s*[:：]?\s*([\d.]+)', 'PaO2'),
            (r'(?:HCO3)\s*[:：]?\s*([\d.]+)', 'HCO3-'),
            (r'(?:BE)\s*[:：]?\s*([\d.-]+)', 'BE'),
            (r'(?:Lac|乳酸)\s*[:：]?\s*([\d.]+)', '乳酸'),
            # 心功能
            (r'(?:BNP)\s*[:：]?\s*([\d.]+)', 'BNP'),
            # BNP + 中文数字（如"BNP三千五"→3500）
            (r'(?:BNP)\s*[:：]?\s*([零一二三四五六七八九十百千万亿]+)', 'BNP_chinese'),
            (r'(?:NT-proBNP)\s*[:：]?\s*([\d.]+)', 'NT-proBNP'),
            # NT-proBNP + 中文数字
            # 胰腺/炎症标志物
            (r'(?:血淀粉酶|淀粉酶)\s*[:：]?\s*([\d.]+)\s*U?/?L?', '血淀粉酶'),
            (r'(?:中性粒细胞|中性粒|中性)\s*[:：]?\s*([\d.]+)\s*%', '中性粒细胞%'),
            # 肝功能
            (r'(?:ALT|谷丙转氨酶)\s*[:：]?\s*([\d.]+)', 'ALT'),
            (r'(?:AST|谷草转氨酶)\s*[:：]?\s*([\d.]+)', 'AST'),
            (r'(?:TBIL|总胆红素)\s*[:：]?\s*([\d.]+)', '总胆红素'),
            (r'(?:DBIL|直接胆红素)\s*[:：]?\s*([\d.]+)', '直接胆红素'),
            (r'(?:ALB|白蛋白)\s*[:：]?\s*([\d.]+)', '白蛋白'),
            # 血常规重点
            (r'(?:WBC|白细胞)\s*[:：]?\s*([\d.]+)\s*[×xX*]\s*10\^?9\s*/\s*L', 'WBC×10^9/L'),
            (r'(?:WBC|白细胞)\s*[:：]?\s*([\d.]+)\s*[×xX*]\s*10\^?9', 'WBC×10^9'),
            (r'(?:WBC|白细胞)\s*[:：]?\s*([\d.]+)', '白细胞'),
            (r'白细胞[高升]?[\s,，]+(\d+(?:\.\d+)?)', '白细胞（高）'),
            (r'(?:RBC|红细胞)\s*[:：]?\s*([\d.]+)', '红细胞'),
            (r'(?:Hb|血红蛋白)\s*[:：]?\s*([\d.]+)', '血红蛋白'),
            (r'(?:PLT|血小板)\s*[:：]?\s*([\d.]+)', '血小板'),]

        
        for pattern, name in lab_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                val = match.group(1)
                # 中文数字转换为阿拉伯数字
                if name.endswith('_chinese'):
                    base_name = name[:-8]  # 去掉"_chinese"后缀
                    int_val = chinese_num_to_arabic(val)
                    pairs.append(f"{base_name} {int_val}")
                elif name.startswith('WBC×10^9'):
                    unit = name.split('×10^9')[1] if '×10^9' in name else ''
                    pairs.append(f"WBC {val}×10^9{unit}")
                else:
                    pairs.append(f"{name} {val}")
        
        if pairs:
            # 去重：同名保留最详细版本（如"WBC 12×10^9/L"优先于"白细胞 12"）
            seen = {}
            for p in pairs:
                pname = p.split()[0] if ' ' in p else p
                if pname not in seen or len(p) > len(seen[pname]):
                    seen[pname] = p
            return "；".join(seen.values())
        return ""
    
    @classmethod
    def extract_medications(cls, text: str) -> str:
        """提取用药信息（分段器+专用提取器方案）"""
        segments = cls.segment_followup_input(text)
        if segments.get("medication"):
            return cls.extract_medication_segment(segments["medication"])
        
        # 降级：旧版正则匹配
        med_pattern = r'(?:吃|服用|用|打)(?:了)?(.+?)(?:[，。；]|$)'
        match = re.search(med_pattern, text)
        if match:
            content = match.group(1).strip()
            medical_indicators = ['mg', 'g', 'ml', '单位', '片', '粒', '次', '天', '每日', 'tid', 'bid', 'qd']
            if any(ind in content for ind in medical_indicators):
                return "目前在" + match.group(0)
        
        return ""
    
    @staticmethod
    def segment_followup_input(text: str) -> dict:
        """
        按固定锚点拆分随访输入文本。
        返回: {"header": "...", "symptoms": "...", "labs": "...", "medication": "..."}
        """
        segments = {"header": "", "symptoms": "", "labs": "", "medication": ""}
        
        # 锚点定义：段名 → 前缀列表
        anchors = {
            "medication": ["目前用药", "用药：", "服药：", "现服"],
            "labs": ["血压", "血糖", "肌酐", "TSH", "CEA", "血氧", "血沉", "CRP", "ALT", "AST", "HBV-DNA"],
            "symptoms": ["今日", "目前症状", "主诉"],
        }
        
        # 找到所有锚点位置
        positions = []
        for seg_name, prefixes in anchors.items():
            for prefix in prefixes:
                idx = text.find(prefix)
                if idx != -1:
                    # 对于"目前用药"需要加冒号才是完整锚点
                    if prefix == "目前用药" and text[idx+4:idx+5] in ['：', ':']:
                        positions.append((idx, seg_name, prefix + text[idx+4]))
                    elif prefix.endswith('：') or prefix.endswith(':'):
                        positions.append((idx, seg_name, prefix))
                    elif prefix in ["今日", "目前症状", "主诉"]:
                        positions.append((idx, seg_name, prefix))
                    elif prefix in ["血压", "血糖", "肌酐", "TSH", "CEA", "血氧", "血沉", "CRP", "ALT", "AST", "HBV-DNA"]:
                        # 检验项锚点：前面是句号/空格/开头，后面是数字
                        if idx == 0 or text[idx-1] in ['。', '，', ' ', '']:
                            positions.append((idx, seg_name, prefix))
        
        # 去重：同一段名只保留最早出现的锚点
        seen_segments = {}
        for pos, seg_name, prefix in sorted(positions):
            if seg_name not in seen_segments:
                seen_segments[seg_name] = (pos, seg_name, prefix)
        
        # 按位置排序
        sorted_segments = sorted(seen_segments.values(), key=lambda x: x[0])
        
        # 切分文本
        for i, (pos, seg_name, prefix) in enumerate(sorted_segments):
            # 当前段内容：从前一个锚点结束到当前锚点开始
            if i == 0:
                start = 0
            else:
                start = sorted_segments[i-1][0] + len(sorted_segments[i-1][1])
                # 跳过前面的"，。"等分隔符
                while start < len(text) and text[start] in ['，', '。', ' ']:
                    start += 1
            
            end = pos
            # 找到当前段的结尾（下一个锚点前）
            if i < len(sorted_segments) - 1:
                end = sorted_segments[i+1][0]
            else:
                end = len(text)
            
            content = text[start:end].strip('，。； ')
            
            # header特殊处理：只取随访类型部分
            if seg_name == "header":
                header_end = content.find('，')
                if header_end != -1:
                    content = content[:header_end]
            
            segments[seg_name] = content
        
        # 如果没找到任何锚点，整个文本作为header
        if not any(segments.values()):
            segments["header"] = text
        
        return segments
    
    @staticmethod
    def extract_medication_segment(med_text: str) -> str:
        """
        用药段专用提取器。
        输入: "目前用药：阿司匹林100mg qd，阿托伐他汀20mg qn"
        输出: "阿司匹林100mg qd，阿托伐他汀20mg qn" 或 "暂无用药"
        """
        if not med_text:
            return ""
        
        # 去掉前缀
        for prefix in ["目前用药：", "目前用药:", "用药：", "用药:", "服药：", "服药:"]:
            if prefix in med_text:
                med_text = med_text.split(prefix, 1)[1]
                break
        
        content = med_text.strip("。；， ")
        
        # 特殊值处理
        no_med_keywords = ["无需化疗", "无", "暂无", "未用药", "无需用药", "停药", "已停药"]
        if content in no_med_keywords or any(kw == content for kw in no_med_keywords):
            return "暂无用药"
        
        # 简单清洗：去掉多余空格
        return content if content else ""
    
    @classmethod
    def extract_pain_location(cls, text: str) -> str:
        """提取疼痛定位信息"""
        patterns = [
            r'(右下腹|左下腹|右上腹|左上腹|上腹部|下腹部|脐周|全腹|左胸|右胸|前胸|后背|左肩|右肩|左膝|右膝|左腕|右腕|左踝|右踝)',
            r'(腹部|胸部|头部|腰部|背部|四肢|关节)',]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return ""
    
    @classmethod
    def extract_with_regex(cls, text: str) -> ExtractedInfo:
        """使用正则表达式提取信息"""
        info = ExtractedInfo()
        
        # 预处理：将中文数字转换为阿拉伯数字（BNP三千五→BNP3500）
        text = cls._preprocess_chinese_numbers(text)
        
        # 提取性别
        for pattern in cls.PATTERNS["gender"]:
            match = re.search(pattern, text)
            if match:
                info.gender = match.group(1)
                break
        if not info.gender and re.search(r'孕妇|产妇', text):
            info.gender = "女"
        if not info.gender:
            kinship_male = re.search(r'其子|其儿|他儿子|她儿子|他儿|她儿|孙子|外孙|儿子|侄子|外甥|我爸|我父亲|我爸爸|我爸', text)
            kinship_female = re.search(r'其女|女儿|孙女|外孙女|侄女|外甥女|他女儿|她女儿|她儿|我妈|我母亲|我妈妈|我妈', text)
            if kinship_male:
                info.gender = "男"
            elif kinship_female:
                info.gender = "女"
        if not info.gender:
            # 老张/老王/老李 等称呼默认男性
            if re.search(r'老[张王李赵刘陈杨黄周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩]:', text) or re.search(r'^老[张王李赵刘陈杨黄周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩]', text):
                info.gender = "男"
        
        # 提取年龄
        for pattern in cls.PATTERNS["age"]:
            match = re.search(pattern, text)
            if match:
                age_val = match.group(1)
                info.age = age_val + "岁"
                break
        
        # 提取主诉（只提取症状+时长模式）
        symptom_pattern = cls.PATTERNS["chief_complaint"][0]
        match = re.search(symptom_pattern, text)
        if match:
            symptom_part = match.group(1)
            valid_symptoms = ['头痛', '胸痛', '腹痛', '发热', '咳嗽', '头晕', '乏力', 
                             '恶心', '呕吐', '呼吸困难', '心悸', '腹泻', '便秘',
                             '关节痛', '腰痛', '视力模糊', '听力下降', '鼻出血', 
                             '牙龈出血', '皮肤瘙痒', '水肿', '麻木', '抽搐', '昏迷', 
                             '意识模糊', '食欲不振', '体重下降', '口干', '多饮', '多尿', 
                             '失眠', '记忆减退', '情绪低落', '胸闷', '气短', '腹胀', 
                             '黄疸', '皮疹', '寒战', '出汗', '面色苍白', '出冷汗',
                             '胸闷', '疼痛']
            if symptom_part in valid_symptoms:
                info.chief_complaint = symptom_part + match.group(2)
        
        # 提取血糖（多值，含校验）
        bs_results = MedicalValueValidator.extract_all_blood_sugars(text)
        if bs_results:
            info.blood_sugar = MedicalValueValidator.format_blood_sugars(bs_results)
        else:
            # 降级：单值模式
            for pattern in cls.PATTERNS["blood_sugar"]:
                match = re.search(pattern, text)
                if match:
                    info.blood_sugar = match.group(1) + " mmol/L"
                    break
        
        # 提取餐后血糖
        for pattern in cls.PATTERNS["blood_sugar_2h"]:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 2 and groups[1]:
                    info.blood_sugar_2h = groups[1] + " mmol/L"
                elif len(groups) == 1:
                    info.blood_sugar_2h = groups[0] + " mmol/L"
                break
        
        # 提取糖化血红蛋白
        for pattern in cls.PATTERNS["hba1c"]:
            match = re.search(pattern, text)
            if match:
                info.hba1c = match.group(1) + "%"
                break
        
        # 提取血压（多值，含校验）
        bp_results = MedicalValueValidator.extract_all_blood_pressures(text)
        if bp_results:
            info.blood_pressure = MedicalValueValidator.format_blood_pressures(bp_results)
        else:
            # 降级：单值模式
            for pattern in cls.PATTERNS["blood_pressure"]:
                match = re.search(pattern, text)
                if match:
                    groups = match.groups()
                    if len(groups) >= 2:
                        info.blood_pressure = f"{groups[0]}/{groups[1]} mmHg"
                    break
        
        # 带否定词的属性提取
        positive, negative = cls.extract_symptoms_safe(text)
        info.positive_symptoms = positive
        info.negative_symptoms = negative
        
        # 传统症状列表（向后兼容）
        all_symptoms = set()
        for pattern in cls.PATTERNS["symptoms"]:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    all_symptoms.update(m for m in match if m)
                else:
                    all_symptoms.add(match)
        info.symptoms = list(all_symptoms)
        
        # 提取心率
        for pattern in cls.PATTERNS["heart_rate"]:
            match = re.search(pattern, text)
            if match:
                info.heart_rate = match.group(1) + "次/分"
                break
        
        # 提取呼吸
        for pattern in cls.PATTERNS["respiratory_rate"]:
            match = re.search(pattern, text)
            if match:
                info.respiratory_rate = match.group(1) + "次/分"
                break
        
        # 提取体温（多值，含校验）
        temp_results = MedicalValueValidator.extract_all_temperatures(text)
        if temp_results:
            info.temperature = MedicalValueValidator.format_temperatures(temp_results)
        else:
            # 降级：单值模式
            for pattern in cls.PATTERNS["temperature"]:
                match = re.search(pattern, text)
                if match:
                    info.temperature = match.group(1) + "℃"
                    break
        
        # 提取检验指标（使用 LAB_PATTERNS）
        lab_field_map = {
            "wbc": "wbc", "neutrophil_pct": "neutrophil_pct", "amylase": "amylase",
            "bnp_value": "bnp", "alt": "alt", "ast": "ast",
            "afp": "afp", "cea": "cea", "ca199": "ca199", "ca153": "ca153",
            "creatinine": "creatinine", "bun": "bun",
            "hbv_dna": "hbv_dna", "esr": "esr", "crp": "crp",
            "pt": "pt", "aptt": "aptt", "fib": "fib", "d_dimer": "d_dimer",
            "tc": "tc", "tg": "tg", "ldl_c": "ldl_c", "hdl_c": "hdl_c", "uric_acid": "uric_acid",
            "tsh": "tsh", "ft3": "ft3", "ft4": "ft4",
            "blood_oxygen": "blood_oxygen",
        }
        for pattern_key, field_name in lab_field_map.items():
            patterns = cls.PATTERNS.get(pattern_key, [])
            if not patterns or getattr(info, field_name, ""):
                continue
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    val = match.group(1)
                    setattr(info, field_name, val)
                    break
        
        # 提取既往史
        history_set = set()
        
        # 提取"既往有XX"格式的内容（支持多个）
        past_history_pattern = r'(既往有[\u4e00-\u9fa5\d年]+?)(，|。|；|$)'
        matches = re.findall(past_history_pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                history_set.add(match[0])
        
        for item in list(history_set):
            if '，' in item:
                history_set.remove(item)
                parts = item.split('，')
                for part in parts:
                    if part:
                        history_set.add(part)
        
        # 提取疾病名+X年模式（作为既往史和潜在诊断来源）
        disease_years_patterns = [
            r'(高血压\d+年)',
            r'(糖尿病\d+年)',
            r'(冠心病\d+年)',
            r'(心脏病\d+年)',
            r'(脑梗死\d+年)',
            r'(脑出血\d+年)',]

        for pattern in disease_years_patterns:
            match = re.search(pattern, text)
            if match:
                disease = match.group(1)
                already_exists = any(disease in h or h in disease for h in history_set)
                if not already_exists:
                    history_set.add(disease)
        
        # 提取"有XX病X年"格式
        has_disease_pattern = r'(有[\u4e00-\u9fa5]+?病\d+年)'
        match = re.search(has_disease_pattern, text)
        if match:
            history_set.add(match.group(1))
        
        # 提取"有高血压"等直接表达的病史（无"病X年"后缀）
        simple_disease_patterns = [
            r'有(高血压)([，。；]|$)',
            r'有(糖尿病)([，。；]|$)',
            r'有(冠心病)([，。；]|$)',
            r'有(心脏病)([，。；]|$)',
            r'有(哮喘)([，。；]|$)',
            r'有(肝炎)([，。；]|$)',
            r'有(肾炎)([，。；]|$)',
            r'有(支气管炎)([，。；]|$)',]

        for pattern in simple_disease_patterns:
            match = re.search(pattern, text)
            if match:
                disease = match.group(1)
                already_exists = any(disease in h or h in disease for h in history_set)
                if not already_exists:
                    history_set.add(disease)
        
        # 提取"既往体健/既往健康"
        if not history_set:
            healthy_pattern = r'(既往体健|既往健康|平素体健|既往一般情况可)'
            match = re.search(healthy_pattern, text)
            if match:
                history_set.add(match.group(1))
        
        info.medical_history = list(history_set)
        
        # 提取影像检查
        imaging_patterns = [
            r'(胸片示[^\，。；]+)',
            r'(胸部CT示[^\，。；]+)',
            r'(头颅CT示[^\，。；]+)',
            r'(头颅MRI示[^\，。；]+)',
            r'(腹部B超示[^\，。；]+)',
            r'(心脏彩超示[^\，。；]+)',
            r'(超声示[^\，。；]+)',
            r'(胸片[：:][^\，。；]+)',
            r'(胸部CT[：:][^\，。；]+)',
            r'(胸片.*?(?:斑片|阴影|渗出|实变|结节|钙化|纤维化)[^\，。；]*)',
            r'(X线.*?(?:斑片|阴影|渗出|实变|结节)[^\，。；]*)',]

        for pattern in imaging_patterns:
            match = re.search(pattern, text)
            if match:
                info.imaging_exam.append(match.group(1))
        
        # 提取发热温度(最高XX℃)
        for pattern in cls.PATTERNS["fever_temp"]:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                for g in groups:
                    if g and g.replace('.', '').isdigit():
                        info.fever_temp = g + "℃"
                        break
                if info.fever_temp:
                    break
        
        # 提取查体和检验原文
        physical_raw, lab_raw = cls.extract_struct_exams(text)
        info.physical_exam_raw = physical_raw
        info.lab_exam_raw = lab_raw
        
        # 提取用药信息
        info.medications = cls.extract_medications(text)
        
        # 提取疼痛定位
        info.pain_location = cls.extract_pain_location(text)
        
        # 提取诊断来源
        for pattern in cls.PATTERNS["diagnosis_source"]:
            match = re.search(pattern, text)
            if match:
                info.has_diagnosis_source = True
                info.diagnosis_source = match.group(2)
                break
        
        # 提取手术史（如PCI术后）
        for pattern in cls.PATTERNS["surgery_history"]:
            match = re.search(pattern, text)
            if match:
                surgery = match.group(1)
                if surgery not in info.medical_history:
                    info.medical_history.append(surgery)
        
        # 提取治疗计划
        for pattern in cls.PATTERNS["treatment_plan"]:
            match = re.search(pattern, text)
            if match:
                info.treatment_plan = match.group(0)
                break
        
        # 判断疾病类型（如果历史有明确疾病，也作为disease_type来源）
        if any(k in text for k in ["糖尿病", "血糖", "胰岛素", "二甲双胍"]):
            info.disease_type = "糖尿病"
        elif any(k in text for k in ["高血压", "降压药", "氨氯地平"]):
            info.disease_type = "高血压"
        elif "血压" in text:
            # 血压数值判断：检测是否为低血压
            bp_val = re.search(r'血压\D*(\d+)/(\d+)', text)
            if bp_val:
                sys_bp, dia_bp = int(bp_val.group(1)), int(bp_val.group(2))
                if sys_bp >= 140 or dia_bp >= 90:
                    info.disease_type = "高血压"
                elif sys_bp < 90 or dia_bp < 60:
                    info.disease_type = "低血压"
                # 正常血压不设置disease_type
        elif (("肺炎" in text) or
              ("胸片" in text and any(k in text for k in ["斑片影", "片状影", "浸润影"])) or
              ("肺部" in text and any(k in text for k in ["感染", "炎症", "斑片", "啰音", "实变"]))):
            info.disease_type = "肺炎"
        
        # 提取患者姓名（支持"老张"、"小刘"等口语化称呼）
        for pattern in cls.PATTERNS["patient_name"]:
            match = re.search(pattern, text)
            if match:
                info.patient_name = match.group(1)
                break
        # 口语化姓名提取："老张"、"小刘"等
        if not info.patient_name:
            spoken_name = re.search(r'(老|小)([\u4e00-\u9fa5])', text)
            if spoken_name:
                info.patient_name = spoken_name.group(0)
        
        # 提取联系电话
        for pattern in cls.PATTERNS["phone"]:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                # 多组模式（如带-的手机号：3组 → 拼接）
                info.phone = ''.join(groups)
                # 座机模式（0xxx-xxxxxxx）可能只有2组
                break
        
        # 提取身份证号
        for pattern in cls.PATTERNS["id_card"]:
            match = re.search(pattern, text)
            if match:
                info.id_card = match.group(1)
                break
        
        # 提取地址
        address_patterns = [
            r'(?:住址|地址|家住)[：:](.+?)(?:[，。；]|$)',
            r'(?:住在|家住)([\u4e00-\u9fa5a-zA-Z\d省市县区镇村路街号栋楼单元室]+)',]

        for pattern in address_patterns:
            match = re.search(pattern, text)
            if match:
                info.address = match.group(1).strip()
                break
        
        # 提取家属信息（排除患者本人）
        family_patterns = [
            r'(?:我|他|她)(?:老公|老婆|丈夫|妻子|爱人|儿子|女儿|父亲|母亲|爸爸|妈妈|爸|妈|儿|女)(?:叫|是)([\u4e00-\u9fa5]{2,4})',]

        for pattern in family_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                info.family_members.append(match)
        
        # 新增：提取检验指标
        lab_mapping = [
            ("blood_oxygen", "blood_oxygen"),
            ("tsh", "tsh"),
            ("ft3", "ft3"),
            ("ft4", "ft4"),
            ("cea", "cea"),
            ("ca199", "ca199"),
            ("ca153", "ca153"),
            ("afp", "afp"),
            ("hbv_dna", "hbv_dna"),
            ("creatinine", "creatinine"),
            ("bun", "bun"),
            ("pt", "pt"),
            ("aptt", "aptt"),
            ("fib", "fib"),
            ("d_dimer", "d_dimer"),
            ("esr", "esr"),
            ("crp", "crp"),
            ("alt", "alt"),
            ("ast", "ast"),
            ("tc", "tc"),
            ("tg", "tg"),
            ("ldl_c", "ldl_c"),
            ("hdl_c", "hdl_c"),
            ("uric_acid", "uric_acid"),
            ("wbc", "wbc"),
            ("neutrophil_pct", "neutrophil_pct"),
            ("amylase", "amylase"),
            ("bnp_value", "bnp"),]

        for pattern_key, field_name in lab_mapping:
            for pattern in cls.PATTERNS.get(pattern_key, []):
                match = re.search(pattern, text)
                if match:
                    setattr(info, field_name, match.group(1))
                    break
        
        return info
    
    @classmethod
    def extract_with_llm(cls, text: str) -> ExtractedInfo:
        """使用大模型提取信息"""
        client = OpenAI(
            api_key=get_api_key(),
            base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        )
        
        system_prompt = """你是一个医疗信息提取专家。
请从以下医患对话中提取关键信息，输出JSON格式。
如果信息不存在或不确定，请留空字符串。
不要生成任何不存在的信息。
不要猜测。

chief_complaint提取规则：
1. 格式必须为"症状+时长"，如"头痛3天""剧烈头痛2小时""胸闷1个月，加重3天"
2. 必须保留程度修饰词（剧烈、轻度、重度、阵发性、持续性、突发等）
3. 必须保留时长信息（X天/X小时/X周/X月等）
4. 如果有外伤/手术/复查场景，主诉开头应包含场景，如"车祸伤后意识模糊1小时""乳腺癌术后复查"
5. 不要提取阴性症状（"无XX"中的XX），只提取阳性症状
6. 口语化表达需标准化："浑身不得劲"→"全身不适"，"隐隐作痛好几天"→"腹痛数日"

输出格式：
{
    "gender": "",
    "age": "",
    "chief_complaint": "",
    "symptoms": [],
    "blood_sugar": "",
    "blood_sugar_2h": "",
    "hba1c": "",
    "blood_pressure": "",
    "disease_type": ""
}
"""
        
        try:
            response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"医患对话：{text}"}],

                temperature=0.0,
                max_tokens=500
            )
            
            result = response.choices[0].message.content
            try:
                data = json.loads(result)
                info = ExtractedInfo()
                info.gender = data.get("gender", "")
                info.age = data.get("age", "")
                info.chief_complaint = data.get("chief_complaint", "")
                info.symptoms = data.get("symptoms", [])
                info.blood_sugar = data.get("blood_sugar", "")
                info.blood_sugar_2h = data.get("blood_sugar_2h", "")
                info.hba1c = data.get("hba1c", "")
                info.blood_pressure = data.get("blood_pressure", "")
                info.disease_type = data.get("disease_type", "")
                return info
            except json.JSONDecodeError:
                logger.error(f"LLM输出不是有效JSON: {result}")
                return ExtractedInfo()
        except Exception as e:
            logger.error(f"LLM提取失败: {e}")
            return ExtractedInfo()
    
    @classmethod
    def extract(cls, text: str, skip_llm: bool = False) -> ExtractedInfo:
        """综合提取：正则为主，LLM为辅
        Args:
            text: 输入文本
            skip_llm: 是否跳过LLM（数据模式跳过LLM避免幻觉）
        """
        # 先用正则提取
        regex_info = cls.extract_with_regex(text)
        
        # 如果要求跳过LLM，直接返回正则结果
        if skip_llm:
            return regex_info
        
        # 如果正则提取结果为空或关键信息缺失，使用LLM补充
        if regex_info.is_empty() or not regex_info.chief_complaint:
            llm_info = cls.extract_with_llm(text)
            
            # 合并结果：正则结果优先级更高（避免幻觉）
            result = ExtractedInfo()
            result.gender = regex_info.gender or llm_info.gender
            result.age = regex_info.age or llm_info.age
            
            # 主诉只接受症状+时长格式，不接受检查指标
            chief_complaint = regex_info.chief_complaint or llm_info.chief_complaint
            if chief_complaint:
                # 口语化主诉标准化
                chief_complaint = cls.standardize_chief_complaint(chief_complaint)
                # 检查是否为有效主诉（包含症状关键词）
                symptom_keywords = ['头痛', '胸痛', '腹痛', '发热', '咳嗽', '头晕', '乏力', 
                                   '恶心', '呕吐', '呼吸困难', '心悸', '腹泻', '便秘',
                                   '关节痛', '腰痛', '视力模糊', '听力下降', '水肿', '麻木']
                has_valid_symptom = any(keyword in chief_complaint for keyword in symptom_keywords)
                if has_valid_symptom:
                    result.chief_complaint = chief_complaint
            
            result.symptoms = regex_info.symptoms or llm_info.symptoms
            result.positive_symptoms = regex_info.positive_symptoms or llm_info.positive_symptoms
            result.negative_symptoms = regex_info.negative_symptoms or llm_info.negative_symptoms
            result.blood_sugar = regex_info.blood_sugar or llm_info.blood_sugar
            result.blood_sugar_2h = regex_info.blood_sugar_2h or llm_info.blood_sugar_2h
            result.hba1c = regex_info.hba1c or llm_info.hba1c
            result.blood_pressure = regex_info.blood_pressure or llm_info.blood_pressure
            result.disease_type = regex_info.disease_type or llm_info.disease_type
            result.medical_history = regex_info.medical_history or llm_info.medical_history
            result.imaging_exam = regex_info.imaging_exam or llm_info.imaging_exam
            result.has_diagnosis_source = regex_info.has_diagnosis_source or llm_info.has_diagnosis_source
            result.diagnosis_source = regex_info.diagnosis_source or llm_info.diagnosis_source
            
            # 新增结构化字段
            result.physical_exam_raw = regex_info.physical_exam_raw or llm_info.physical_exam_raw
            result.lab_exam_raw = regex_info.lab_exam_raw or llm_info.lab_exam_raw
            result.medications = regex_info.medications or llm_info.medications
            result.pain_location = regex_info.pain_location or llm_info.pain_location
            
            # 补充遗漏字段
            result.fever_temp = regex_info.fever_temp or llm_info.fever_temp
            result.temperature = regex_info.temperature or llm_info.temperature
            result.heart_rate = regex_info.heart_rate or llm_info.heart_rate
            result.respiratory_rate = regex_info.respiratory_rate or llm_info.respiratory_rate
            result.treatment_plan = regex_info.treatment_plan or llm_info.treatment_plan
            
            # 患者基本信息
            result.patient_name = regex_info.patient_name or llm_info.patient_name
            result.phone = regex_info.phone or llm_info.phone
            result.id_card = regex_info.id_card or llm_info.id_card
            result.address = regex_info.address or llm_info.address
            result.family_members = regex_info.family_members or llm_info.family_members
            
            # 新增检验指标字段（正则提取为主，LLM不覆盖）
            result.wbc = regex_info.wbc
            result.neutrophil_pct = regex_info.neutrophil_pct
            result.amylase = regex_info.amylase
            result.bnp = regex_info.bnp
            result.alt = regex_info.alt
            result.ast = regex_info.ast
            result.blood_oxygen = regex_info.blood_oxygen
            result.tsh = regex_info.tsh
            result.ft3 = regex_info.ft3
            result.ft4 = regex_info.ft4
            result.cea = regex_info.cea
            result.ca199 = regex_info.ca199
            result.ca153 = regex_info.ca153
            result.creatinine = regex_info.creatinine
            result.bun = regex_info.bun
            result.afp = regex_info.afp
            result.hbv_dna = regex_info.hbv_dna
            result.pt = regex_info.pt
            result.aptt = regex_info.aptt
            result.fib = regex_info.fib
            result.d_dimer = regex_info.d_dimer
            result.esr = regex_info.esr
            result.crp = regex_info.crp
            result.alt = regex_info.alt
            result.ast = regex_info.ast
            result.tc = regex_info.tc
            result.tg = regex_info.tg
            result.ldl_c = regex_info.ldl_c
            result.hdl_c = regex_info.hdl_c
            result.uric_acid = regex_info.uric_acid
            result.ca153 = regex_info.ca153
            result.hbv_dna = regex_info.hbv_dna
            
            return result
        
        return regex_info


class FactValidator:
    """事实校验器 - 防止幻觉"""
    
    @classmethod
    def validate_info(cls, info: ExtractedInfo, original_text: str) -> ExtractedInfo:
        """校验提取结果是否在原文中存在"""
        validated = ExtractedInfo()
        
        # 校验性别：原文显式出现，或从亲属/称呼推断
        if info.gender:
            if info.gender in original_text:
                validated.gender = info.gender
            elif re.search(r'老[张王李赵刘陈杨黄周吴徐孙马胡朱郭何罗高林郑梁谢唐许冯宋韩]', original_text):
                validated.gender = info.gender
            elif re.search(r'我爸|我父亲|我爸爸|我妈|我母亲|我妈妈|其子|其女|孙子|孙女|外孙|外孙女', original_text):
                validated.gender = info.gender
        
        # 校验年龄
        if info.age:
            # 检查年龄数字是否在原文中
            age_num = re.search(r'\d+', info.age)
            if age_num and age_num.group() in original_text:
                validated.age = info.age
        
        # 校验主诉（放宽：至少有一个关键词匹配即可，支持API生成内容）
        if info.chief_complaint:
            complaint_parts = re.findall(r'[\u4e00-\u9fa5]+|\d+', info.chief_complaint)
            matched_parts = [p for p in complaint_parts if p in original_text]
            if matched_parts or len(complaint_parts) <= 4:
                validated.chief_complaint = info.chief_complaint
        
        # 校验血糖值
        if info.blood_sugar:
            value = re.search(r'\d+\.?\d*', info.blood_sugar)
            if value and value.group() in original_text:
                validated.blood_sugar = info.blood_sugar
        
        # 校验餐后血糖
        if info.blood_sugar_2h:
            value = re.search(r'\d+\.?\d*', info.blood_sugar_2h)
            if value and value.group() in original_text:
                validated.blood_sugar_2h = info.blood_sugar_2h
        
        # 校验糖化血红蛋白
        if info.hba1c:
            value = re.search(r'\d+\.?\d*', info.hba1c)
            if value and value.group() in original_text:
                validated.hba1c = info.hba1c
        
        # 校验血压
        if info.blood_pressure:
            values = re.findall(r'\d{2,3}', info.blood_pressure)
            if values and all(v in original_text for v in values):
                validated.blood_pressure = info.blood_pressure
        
        # 症状无需严格校验（可能是同义替换）
        validated.symptoms = info.symptoms
        validated.positive_symptoms = info.positive_symptoms
        validated.negative_symptoms = info.negative_symptoms
        
        # 疾病类型基于关键词判断，无需额外校验
        validated.disease_type = info.disease_type
        
        # 发热温度校验
        if info.fever_temp:
            value = re.search(r'\d+\.?\d*', info.fever_temp)
            if value and value.group() in original_text:
                validated.fever_temp = info.fever_temp
        
        # 体温校验
        if info.temperature:
            value = re.search(r'\d+\.?\d*', info.temperature)
            if value and value.group() in original_text:
                validated.temperature = info.temperature
        
        # 心率/呼吸校验
        if info.heart_rate:
            validated.heart_rate = info.heart_rate
        if info.respiratory_rate:
            validated.respiratory_rate = info.respiratory_rate
        
        # 既往史校验
        if info.medical_history:
            validated.medical_history = info.medical_history
        
        # 影像检查校验
        if info.imaging_exam:
            validated.imaging_exam = info.imaging_exam
        
        # 诊断来源校验
        validated.has_diagnosis_source = info.has_diagnosis_source
        validated.diagnosis_source = info.diagnosis_source
        
        # 患者基本信息校验
        validated.patient_name = info.patient_name
        validated.phone = info.phone
        validated.id_card = info.id_card
        validated.address = info.address
        validated.family_members = info.family_members
        
        # 新增字段校验
        validated.positive_symptoms = info.positive_symptoms
        validated.negative_symptoms = info.negative_symptoms
        validated.physical_exam_raw = info.physical_exam_raw
        validated.lab_exam_raw = info.lab_exam_raw
        validated.medications = info.medications
        validated.pain_location = info.pain_location
        
        # 新增检验指标字段直接传递（数值已在extract_info中用正则提取，无需额外校验）
        validated.blood_oxygen = info.blood_oxygen
        validated.wbc = info.wbc
        validated.neutrophil_pct = info.neutrophil_pct
        validated.amylase = info.amylase
        validated.bnp = info.bnp
        validated.tsh = info.tsh
        validated.ft3 = info.ft3
        validated.ft4 = info.ft4
        validated.cea = info.cea
        validated.ca199 = info.ca199
        validated.ca153 = info.ca153
        validated.afp = info.afp
        validated.hbv_dna = info.hbv_dna
        validated.creatinine = info.creatinine
        validated.bun = info.bun
        validated.pt = info.pt
        validated.aptt = info.aptt
        validated.fib = info.fib
        validated.d_dimer = info.d_dimer
        validated.esr = info.esr
        validated.crp = info.crp
        validated.alt = info.alt
        validated.ast = info.ast
        validated.tc = info.tc
        validated.tg = info.tg
        validated.ldl_c = info.ldl_c
        validated.hdl_c = info.hdl_c
        validated.uric_acid = info.uric_acid
        validated.fever_temp = info.fever_temp
        validated.temperature = info.temperature
        
        return validated


class TemplateFiller:
    """模板填充器 - 将提取的信息填入标准模板"""
    
    COLLOQUIAL_MARKERS = ['浑身', '不得劲', '不舒服', '吃饭不香', '睡觉不好',
                          '没劲', '不舒服', '不得劲儿', '哪儿', '说不上来']
    
    @classmethod
    def is_colloquial(cls, text: str) -> bool:
        """检测是否为口语化描述（基于原文，不依赖已提取的info）"""
        has_colloquial_marker = any(m in text for m in cls.COLLOQUIAL_MARKERS)
        return has_colloquial_marker
    
    @classmethod
    def extract_colloquial_description(cls, text: str) -> str:
        """从口语中提取主诉描述"""
        # 去掉口语填充词
        fillers = ['那个', '这个', '就是', '吧', '嗯', '啊', '呢', '呗', '嘛',
                   '哦', '哈', '呀']
        clean = text
        for filler in fillers:
            clean = clean.replace(filler, '')
        # 去掉"你看着给查查吧"等结尾客套话
        clean = re.sub(r'你看着[^。？]*', '', clean)
        clean = re.sub(r'你给[^。？]*', '', clean)
        # 去掉开头的亲属指代（"我爸，他，" → 去掉）
        clean = re.sub(r'^[，,\s]*我(?:爸|妈|哥|姐|弟|妹|爷爷|奶奶|外公|外婆)[，,\s]*他[，,\s]*', '', clean)
        # 去掉多余逗号
        clean = re.sub(r'，{2,}', '，', clean)
        clean = clean.strip('，。！？')
        return clean
    
    @classmethod
    
    @staticmethod
    def _build_present_illness_narrative(parts: list) -> str:
        """将present_illness_parts转为流畅医学叙述"""
        if not parts:
            return "待补充"
        # Filter out "辅助检查结果如下" - exam info goes in 辅助检查 section
        parts = [p for p in parts if p != "辅助检查结果如下"]
        # Deduplicate generic symptoms when specific ones exist
        generic_set = {"疼痛", "不适", "异常"}
        has_specific = any("隐痛" in p or "刺痛" in p or "钝痛" in p or "胀痛" in p or "绞痛" in p for p in parts)
        if has_specific:
            parts = [p for p in parts if not any(g in p.split("：")[-1] for g in generic_set if g in p.split("：")[-1] and len(p.split("：")[-1]) <= 2)]
        
        result = ""
        for i, p in enumerate(parts):
            p = p.replace("阳性症状：", "伴")
            p = p.replace("阴性症状：", "，")
            # 过滤：去除"吞咽困难""黑便"等在对话中医生问诊词（非患者主诉）
            p = p.replace("伴吞咽困难", "").replace("伴黑便", "")
            if i > 0 and not p.startswith("，") and not p.startswith("。"):
                result += "，"
            result += p
        result = result.replace("，，", "，")
        result = result.strip("，")
        return result

    @classmethod
    def _generate_treatment_plan(cls, diagnosis_list: list, disease_type: str, symptoms: list, raw_input: str) -> str:
        """基于诊断和症状生成保守治疗建议（仅供医生参考）"""
        if not diagnosis_list or diagnosis_list == ["待补充"]:
            return "待补充"
        
        # 合并诊断文本用于关键词匹配
        diag_text = " ".join(str(d) for d in diagnosis_list) + " " + (disease_type or "")
        diag_text_lower = diag_text.lower()
        raw_lower = raw_input.lower() if raw_input else ""
        
        plans = []
        
        # 感染/炎症类 → 抗感染 + 对症
        if any(k in diag_text for k in ["肺炎", "支气管炎", "感染", "阑尾炎", "胆囊炎", "胰腺炎",
                                          "胃肠炎", "肾盂肾炎", "盆腔炎", "蜂窝织炎", "丹毒"]):
            plans.append("抗感染治疗")
            plans.append("密切监测生命体征")
        
        # 高血压
        if any(k in diag_text for k in ["高血压"]):
            plans.append("监测血压，调整降压药物")
            plans.append("低盐低脂饮食")
        
        # 糖尿病
        if any(k in diag_text for k in ["糖尿病"]):
            plans.append("监测血糖，调整降糖方案")
            plans.append("糖尿病饮食指导")
        
        # 疼痛类
        if any(k in diag_text for k in ["头痛", "偏头痛", "腹痛", "胸痛", "关节痛", "腰痛",
                                          "痛风", "痛经", "三叉神经痛", "带状疱疹"]):
            plans.append("对症止痛治疗")
        
        # 发热：仅当输入明确描述发热（非医生问诊），或体温数值异常
        has_fever_in_input = any(k in raw_lower for k in ["发热", "发烧"]) and not any(
            neg in raw_lower for neg in ["有没有发热", "有没有发烧", "不发热", "不发烧", "无发热", "无发烧"])
        has_temp_value = bool(re.search(r'(\d{2,3})\s*[℃°度]', raw_input)) and not re.search(r'体温\s*正常', raw_input)
        if has_fever_in_input or has_temp_value or any(k in diag_text for k in ["发热"]):
            if "退热" not in " ".join(plans):
                plans.append("必要时退热治疗")
        
        # 外伤/骨折
        if any(k in diag_text for k in ["外伤", "骨折", "挫伤", "扭伤", "脱位", "软组织"]):
            plans.append("伤口处理/固定制动")
            plans.append("必要时X线/CT检查")
        
        # 过敏类
        if any(k in diag_text for k in ["过敏", "荨麻疹", "湿疹", "皮炎", "药疹"]):
            plans.append("抗过敏治疗")
            plans.append("避免接触过敏原")
        
        # 消化系统
        if any(k in diag_text for k in ["胃炎", "溃疡", "反流", "消化不良"]):
            if "抗感染" not in " ".join(plans):
                plans.append("抑酸护胃治疗")
            plans.append("饮食指导")
        
        # 呼吸系统
        if any(k in diag_text for k in ["哮喘", "慢阻肺", "COPD"]):
            plans.append("支气管扩张剂吸入")
            plans.append("必要时氧疗")
        
        # 上呼吸道感染
        if any(k in diag_text for k in ["上呼吸道感染", "感冒", "咽炎", "扁桃体炎"]):
            plans.append("对症支持治疗")
            plans.append("多饮水，注意休息")
        
        # 神经系统
        if any(k in diag_text for k in ["眩晕", "头晕"]):
            plans.append("对症止晕治疗")
            plans.append("卧床休息，避免体位骤变")
        
        if any(k in diag_text for k in ["脑梗", "脑出血", "TIA", "脑卒中", "中风"]):
            plans.append("神经内科紧急评估")
            plans.append("控制血压，抗血小板/抗凝治疗")
        
        # 贫血
        if any(k in diag_text for k in ["贫血"]):
            plans.append("明确贫血病因")
            plans.append("必要时补充铁剂/维生素B12")
        
        # 心律失常
        if any(k in diag_text for k in ["心律失常", "房颤", "房扑", "早搏", "心动过速", "心动过缓"]):
            plans.append("心内科评估")
            plans.append("必要时抗心律失常/抗凝治疗")
        
        # 心衰
        if any(k in diag_text for k in ["心衰", "心力衰竭"]):
            plans.append("限制水钠摄入")
            plans.append("利尿、强心等综合治疗")
            plans.append("心内科住院治疗")
        
        # 甲状腺
        if any(k in diag_text for k in ["甲亢", "甲减", "甲状腺"]):
            plans.append("内分泌科评估")
            plans.append("监测甲功，调整药物")
        
        # 肝功能异常
        if any(k in diag_text for k in ["肝炎", "肝硬化", "肝损伤", "脂肪肝"]):
            plans.append("保肝治疗")
            plans.append("戒酒，避免肝毒性药物")
        
        # 肾功能异常
        if any(k in diag_text for k in ["肾炎", "肾病", "肾衰竭", "肾功能"]):
            plans.append("肾内科评估")
            plans.append("控制血压，低盐低蛋白饮食")
        
        # 肿瘤类
        if any(k in diag_text for k in ["癌", "肿瘤", "恶性", "占位"]):
            plans.append("肿瘤科/相关专科进一步评估")
            plans.append("完善影像学及病理检查")
        
        # 通用兜底
        if not plans:
            plans.append("进一步检查明确诊断")
            plans.append("对症支持治疗")
        else:
            # 总是加上观察建议
            if "进一步检查" not in " ".join(plans) and "评估" not in " ".join(plans):
                plans.append("必要时进一步检查")
        
        return "；".join(plans)


    def fill_admission_note(cls, info: ExtractedInfo, input_type: str = "dialogue", raw_input: str = "") -> MedicalRecord:
        """填充入院记录模板"""
        logger.info(f"FILL_ADM: disease_type={info.disease_type!r}, has_diag={bool(info.disease_type)}, input_type={input_type}")
        
        # 检测是否为对话格式
        is_dialogue = detect_dialogue_format(raw_input)
        
        record = MedicalRecord(record_type=MedicalRecordType.ADMISSION_NOTE)
        
        # 填充患者基本信息（使用字段级脱敏）
        name = PrivacyDesensitizer.desensitize_value('name', info.patient_name) if info.patient_name else "待补充"
        gender = info.gender if info.gender else "待补充"
        age = info.age if info.age else "待补充"
        phone = PrivacyDesensitizer.desensitize_value('phone', info.phone) if info.phone else "待补充"
        id_card = PrivacyDesensitizer.desensitize_value('id_card', info.id_card) if info.id_card else "待补充"
        address = PrivacyDesensitizer.desensitize_value('address', info.address) if info.address else ""
        
        patient_info_lines = []
        patient_info_lines.append("**患者基本信息：**")
        patient_info_lines.append(f"- 姓名：{name}")
        patient_info_lines.append(f"- 性别：{gender}")
        patient_info_lines.append(f"- 年龄：{age}")
        patient_info_lines.append(f"- 联系电话：{phone}")
        patient_info_lines.append(f"- 身份证号：{id_card}")
        if address:
            patient_info_lines.append(f"- 地址：{address}")
        record.patient_info = "".join(patient_info_lines)
        
        # 判断是否存在明确的既往诊断（如"高血压20年"暗示既往已确诊）
        has_explicit_disease_history = bool(info.medical_history)
        
        # 根据输入类型处理主诉
        if input_type == "data":
            record.chief_complaint = "患者提供检验数据"
            record.present_illness = "详见辅助检查"
            # 纯数据模式：覆盖疾病类型为"需结合临床表现"
            record.disease_type = "待补充"
        elif input_type == "unknown":
            record.chief_complaint = "未能识别有效医学信息，请输入中文描述患者病情"
            record.present_illness = "待补充"
        elif input_type == "mixed":
            # 过滤：确保阳性症状不含否定表述，防止"无发热"被当作阳性症状
            filtered_positive = []
            for s in info.positive_symptoms:
                if re.search(r'^(无|否认|没有|未|不).*', s):
                    if s not in info.negative_symptoms:
                        info.negative_symptoms.append(s)
                else:
                    filtered_positive.append(s)

            _is_all_negative = not filtered_positive and bool(info.negative_symptoms) and not info.symptoms
            if _is_all_negative:
                record.chief_complaint = "待补充（无明确症状）"
                present_illness_parts = []
                if info.gender and info.age:
                    present_illness_parts.append(f"患者{info.gender}，{info.age}")
                if info.negative_symptoms:
                    neg_with_prefix = ['无' + s for s in info.negative_symptoms]
                    present_illness_parts.append(f"阴性症状：{'、'.join(neg_with_prefix)}")
                record.present_illness = cls._build_present_illness_narrative(present_illness_parts)
            else:
                if info.chief_complaint:
                    record.chief_complaint = info.chief_complaint
                elif filtered_positive:
                    record.chief_complaint = "、".join(filtered_positive)[:30]
                elif info.symptoms:
                    record.chief_complaint = "、".join(info.symptoms)[:20] + "待查"
                else:
                    record.chief_complaint = "患者提供症状描述及检验数据"
                
                present_illness_parts = []
                if record.chief_complaint == "患者提供检验数据":
                    record.present_illness = "详见辅助检查"
                    if not info.disease_type or info.disease_type == "待补充":
                        info.disease_type = "待补充"
                if info.gender and info.age:
                    present_illness_parts.append(f"患者{info.gender}，{info.age}")
                if record.chief_complaint:
                    present_illness_parts.append(f"因\"{record.chief_complaint}\"就诊")
                
                if filtered_positive and record.chief_complaint:
                    unique_positive = [s for s in filtered_positive if s not in record.chief_complaint]
                    if unique_positive:
                        present_illness_parts.append(f"阳性症状：{'、'.join(unique_positive)}")
                
                if info.negative_symptoms:
                    neg_with_prefix = ['无' + s for s in info.negative_symptoms]
                    present_illness_parts.append(f"阴性症状：{'、'.join(neg_with_prefix)}")
                
                if info.fever_temp:
                    present_illness_parts.append(f"发热最高{info.fever_temp}")
                
                if info.pain_location:
                    present_illness_parts.append(f"位于{info.pain_location}")
                
                if info.medications:
                    present_illness_parts.append(info.medications)
                
                if record.chief_complaint != "患者提供检验数据":
                    has_exam_data = info.lab_exam_raw or info.imaging_exam or info.blood_sugar or info.blood_pressure
                    if has_exam_data:
                        present_illness_parts.append("辅助检查结果如下")
                    record.present_illness = cls._build_present_illness_narrative(present_illness_parts)
        else:
            # 对话模式：正常处理
            # 过滤：确保阳性症状不含否定表述
            filtered_positive = []
            for s in info.positive_symptoms:
                if re.search(r'^(无|否认|没有|未|不).*', s):
                    if s not in info.negative_symptoms:
                        info.negative_symptoms.append(s)
                else:
                    filtered_positive.append(s)

            _is_all_negative = not filtered_positive and bool(info.negative_symptoms) and not info.symptoms
            if _is_all_negative:
                record.chief_complaint = "待补充（无明确症状）"
                present_illness_parts = []
                if info.gender and info.age:
                    present_illness_parts.append(f"患者{info.gender}，{info.age}")
                if info.negative_symptoms:
                    neg_with_prefix = ['无' + s for s in info.negative_symptoms]
                    present_illness_parts.append(f"阴性症状：{'、'.join(neg_with_prefix)}")
                record.present_illness = cls._build_present_illness_narrative(present_illness_parts)
            else:
                # 检测是否为口语化描述（基于原文）
                is_colloquial = cls.is_colloquial(raw_input)
                
                if is_colloquial:
                    # 口语输入：优先使用API生成的规范主诉
                    if info.chief_complaint and len(info.chief_complaint) >= 4 and '待查' not in info.chief_complaint and '未能' not in info.chief_complaint:
                        record.chief_complaint = info.chief_complaint
                        record.present_illness = f"患者自述：{raw_input[:100]}"
                    else:
                        # 保留原话风格
                        colloquial_desc = cls.extract_colloquial_description(raw_input)
                        main_symptoms = colloquial_desc
                        parts = [p.strip() for p in colloquial_desc.split('，') if p.strip()]
                        symptom_parts = [p for p in parts if any(s in p for s in ['不得劲', '不香', '不好', '不舒服', '难受', '疼', '痛', '热', '咳', '晕'])]
                        if symptom_parts:
                            main_symptoms = '、'.join(symptom_parts)
                        elif parts:
                            main_symptoms = '、'.join(parts[-3:])
                        record.chief_complaint = main_symptoms[:30] if len(main_symptoms) > 30 else main_symptoms
                        record.present_illness = f"患者自述：{colloquial_desc}"
                elif info.chief_complaint:
                    record.chief_complaint = info.chief_complaint
                elif filtered_positive:
                    record.chief_complaint = "、".join(filtered_positive)[:30]
                elif info.symptoms:
                    record.chief_complaint = "、".join(info.symptoms)[:20] + "待查"
                else:
                    record.chief_complaint = "待补充"
                
                if not is_colloquial:
                    present_illness_parts = []
                    if info.gender and info.age:
                        present_illness_parts.append(f"患者{info.gender}，{info.age}")
                    elif info.gender:
                        present_illness_parts.append(f"患者{info.gender}")
                    elif info.age:
                        present_illness_parts.append(f"患者{info.age}")
                    
                    if record.chief_complaint:
                        present_illness_parts.append(f"因\"{record.chief_complaint}\"入院")
                    
                    # 阳性症状
                    if filtered_positive and record.chief_complaint:
                        unique_positive = [s for s in filtered_positive if s not in record.chief_complaint]
                        if unique_positive:
                            present_illness_parts.append(f"阳性症状：{'、'.join(unique_positive)}")
                    
                    # 阴性症状
                    if info.negative_symptoms:
                        neg_with_prefix = ['无' + s for s in info.negative_symptoms]
                        present_illness_parts.append(f"阴性症状：{'、'.join(neg_with_prefix)}")
                    
                    if info.fever_temp:
                        present_illness_parts.append(f"发热最高{info.fever_temp}")
                    
                    if info.pain_location:
                        present_illness_parts.append(f"位于{info.pain_location}")
                    
                    if info.medications:
                        present_illness_parts.append(info.medications)
                    
                    record.present_illness = cls._build_present_illness_narrative(present_illness_parts)
        
        # 主诉清理：去除换行符和多余空白
        if record.chief_complaint:
            record.chief_complaint = ' '.join(record.chief_complaint.split())
        
        # 体格检查（标准格式）
        physical_exam_lines = []
        
        # 对话格式：从原始文本中提取生命体征
        if is_dialogue:
            vitals = extract_vitals_from_text(raw_input)
            gender_part = info.gender if info.gender else "待补充"
            age_part = info.age.replace('岁', '') if info.age else "待补充"
            physical_exam_lines.append(f"- 一般情况：{gender_part}，{age_part}岁")
            
            t = vitals.get('temperature', '')
            bp = vitals.get('bp', '')
            hr = vitals.get('hr', '')
            rr = vitals.get('rr', '')
            spo2 = vitals.get('spo2', '')
            
            t_display = f"T{t}℃" if t else ""
            p_display = f"P{hr}次/分" if hr else ""
            r_display = f"R{rr}次/分" if rr else ""
            bp_clean = bp.replace('BP', '').replace('bp', '').strip() if bp else bp
            bp_display = f"BP{bp_clean}" if bp_clean else ""
            vs_items = [x for x in [t_display, p_display, r_display, bp_display] if x]
            vs_display = "，".join(vs_items) if vs_items else "待补充"
            physical_exam_lines.append(f"- 生命体征：{vs_display}")
        else:
            gender_part = info.gender if info.gender else "待补充"
            age_part = info.age.replace('岁', '') if info.age else "待补充"
            physical_exam_lines.append(f"- 一般情况：{gender_part}，{age_part}岁")
            
            temp_part_raw = info.temperature if info.temperature else "待补充"
            pulse_part = info.heart_rate.replace('次/分', '') if info.heart_rate else "待补充"
            resp_part = info.respiratory_rate.replace('次/分', '') if info.respiratory_rate else "待补充"
            bp_part_raw = info.blood_pressure if info.blood_pressure else "待补充"
            if '，' in temp_part_raw or '[异常' in temp_part_raw:
                temp_display = temp_part_raw
            else:
                temp_clean = temp_part_raw.replace('℃', '')
                temp_display = f"T{temp_clean}℃" if temp_clean != "待补充" else "待补充"
            if '，' in bp_part_raw or '[异常' in bp_part_raw:
                bp_display = bp_part_raw
            else:
                bp_clean = bp_part_raw.replace('BP', '').replace('bp', '').strip() if bp_part_raw and bp_part_raw != "待补充" else bp_part_raw
                bp_display = bp_clean if bp_clean and bp_clean != "待补充" else "待补充"
            p_disp = f"P{pulse_part}次/分" if pulse_part != "待补充" else "待补充"
            r_disp = f"R{resp_part}次/分" if resp_part != "待补充" else "待补充"
            vs_items = []
            if temp_display != "待补充": vs_items.append(temp_display)
            if p_disp != "待补充": vs_items.append(p_disp)
            if r_disp != "待补充": vs_items.append(r_disp)
            if bp_display != "待补充": vs_items.append(f"BP{bp_display}")
            vs_display = "，".join(vs_items) if vs_items else "待补充"
            physical_exam_lines.append(f"- 生命体征：{vs_display}")
        
        # 使用结构化查体数据
        if info.physical_exam_raw:
            physical_exam_lines.append(f"- 查体发现：{info.physical_exam_raw}")
        else:
            physical_exam_lines.append("- 其他：待补充")
        
        record.physical_exam = "\n".join(physical_exam_lines)
        
        # 辅助检查
        auxiliary_exam_parts = []
        if info.blood_sugar:
            auxiliary_exam_parts.append(f"空腹血糖：{info.blood_sugar}")
        if info.blood_sugar_2h:
            auxiliary_exam_parts.append(f"餐后2小时血糖：{info.blood_sugar_2h}")
        if info.hba1c:
            auxiliary_exam_parts.append(f"糖化血红蛋白：{info.hba1c}")
        if info.blood_pressure:
            auxiliary_exam_parts.append(f"血压：{info.blood_pressure}")
        if info.heart_rate:
            auxiliary_exam_parts.append(f"心率：{info.heart_rate}")
        # 新增检验指标
        if info.blood_oxygen:
            auxiliary_exam_parts.append(f"血氧：{info.blood_oxygen}%")
        if info.tsh:
            auxiliary_exam_parts.append(f"TSH：{info.tsh}")
        if info.ft3:
            auxiliary_exam_parts.append(f"FT3：{info.ft3}")
        if info.ft4:
            auxiliary_exam_parts.append(f"FT4：{info.ft4}")
        if info.cea:
            auxiliary_exam_parts.append(f"CEA：{info.cea}")
        if info.ca199:
            auxiliary_exam_parts.append(f"CA199：{info.ca199}")
        if info.ca153:
            auxiliary_exam_parts.append(f"CA153：{info.ca153}")
        if info.creatinine:
            auxiliary_exam_parts.append(f"肌酐：{info.creatinine}")
        if info.bun:
            auxiliary_exam_parts.append(f"尿素氮：{info.bun}")
        if info.afp:
            auxiliary_exam_parts.append(f"AFP：{info.afp}")
        if info.hbv_dna:
            auxiliary_exam_parts.append(f"HBV-DNA：{info.hbv_dna}")
        if info.pt:
            auxiliary_exam_parts.append(f"PT：{info.pt}s")
        if info.aptt:
            auxiliary_exam_parts.append(f"APTT：{info.aptt}s")
        if info.fib:
            auxiliary_exam_parts.append(f"FIB：{info.fib}")
        if info.d_dimer:
            auxiliary_exam_parts.append(f"D-二聚体：{info.d_dimer}")
        if info.esr:
            auxiliary_exam_parts.append(f"血沉：{info.esr}")
        if info.crp:
            auxiliary_exam_parts.append(f"CRP：{info.crp}")
        if info.alt:
            auxiliary_exam_parts.append(f"ALT：{info.alt}")
        if info.ast:
            auxiliary_exam_parts.append(f"AST：{info.ast}")
        if info.tc:
            auxiliary_exam_parts.append(f"TC：{info.tc}")
        if info.tg:
            auxiliary_exam_parts.append(f"TG：{info.tg}")
        if info.ldl_c:
            auxiliary_exam_parts.append(f"LDL-C：{info.ldl_c}")
        if info.hdl_c:
            auxiliary_exam_parts.append(f"HDL-C：{info.hdl_c}")
        if info.uric_acid:
            auxiliary_exam_parts.append(f"尿酸：{info.uric_acid}")
        if info.wbc:
            # WBC始终以×10^9/L为单位输出（标准医学格式）
            auxiliary_exam_parts.append(f"WBC：{info.wbc}×10^9/L")
        if info.neutrophil_pct:
            auxiliary_exam_parts.append(f"中性粒细胞：{info.neutrophil_pct}%")
        if info.amylase:
            auxiliary_exam_parts.append(f"血淀粉酶：{info.amylase}")
        if info.bnp:
            auxiliary_exam_parts.append(f"BNP：{info.bnp}")
        
        # 对话格式：从原始文本中提取检验结果
        if is_dialogue:
            dialogue_labs = extract_labs_from_dialogue(raw_input)
            existing_labs = set()
            for part in auxiliary_exam_parts:
                # 提取已有检验中的关键词用于去重
                for key in ['WBC', '白细胞', 'Hb', '血红蛋白', 'PLT', '血小板', 'CRP',
                             'BNP', 'PCT', 'ALT', 'AST', 'Cr', '肌酐', '血糖',
                             'TC', 'TG', 'LDL', 'HDL', 'HCG', '孕酮']:
                    if key in part:
                        existing_labs.add(key)
            for lab in dialogue_labs:
                # 去重：避免与已有检验项重复
                should_add = True
                for ek in existing_labs:
                    if ek in lab:
                        should_add = False
                        break
                if should_add:
                    auxiliary_exam_parts.append(lab)
        
        # 添加检验原文
        if info.lab_exam_raw:
            # 去重：如果已有WBC值，从lab_exam_raw中移除重复部分
            lab_clean = info.lab_exam_raw
            if info.wbc and lab_clean:
                lab_clean = re.sub(r'WBC\s*[\d.]+\s*[；;]?\s*', '', lab_clean)
                lab_clean = re.sub(r'白细胞\s*[\d.]+\s*[；;]?\s*', '', lab_clean)
                lab_clean = lab_clean.strip('；;，, ')
            if lab_clean:
                auxiliary_exam_parts.append(f"检验：{lab_clean}")
        
        # 添加影像检查
        if info.imaging_exam:
            for exam in info.imaging_exam:
                auxiliary_exam_parts.append(f"{exam}")
        
        if auxiliary_exam_parts:
            record.auxiliary_exam = "".join(auxiliary_exam_parts)
        elif input_type == "data":
            # 数据模式下即使结构提取失败，也尝试从原文提取数值
            raw_values = re.findall(r'[\d.]+', raw_input)
            if raw_values:
                record.auxiliary_exam = f"检验数据：{raw_input[:200]}"
            else:
                record.auxiliary_exam = "本次无辅助检查"
        else:
            record.auxiliary_exam = "本次无辅助检查"
        
        # 既往史
        if info.medical_history:
            record.past_history = "、".join(info.medical_history)
        else:
            record.past_history = "未提供"
        
        # 初步诊断（诊断安全校验 - 改进版）
        # 规则：
        # 1. 输入含"诊断为XX"/"考虑XX"/"疑似XX" → "XX（来源：患者自述/外院诊断）"
        # 2. 输入含明确病史（XX病X年）→ "XX（来源：患者自述，病史X年）"
        # 3. 输入含症状+检验数据+疾病关键词 → "XX可能，建议进一步检查"
        # 4. 仅含检验数值 → "待补充"
        # 5. 无任何有效信息 → "待补充"
        
        # 检查是否有明确的疾病史（从medical_history中提取疾病名称）
        disease_from_history = ""
        disease_history_list = []  # 收集全部病史疾病
        if info.medical_history:
            for hist in info.medical_history:
                for disease in ['高血压', '糖尿病', '冠心病', '脑梗死', '脑出血', '肺炎', '哮喘',
                               '慢性阻塞性肺疾病', '心房颤动', '心力衰竭', '慢性肾炎', '肝炎',
                               '肝硬化', '甲状腺功能亢进', '甲状腺功能减退', '类风湿关节炎']:
                    if disease in hist:
                        year_match = re.search(r'(\d+)年', hist)
                        years = f"病史{year_match.group(1)}年" if year_match else "既往病史"
                        disease_history_list.append((disease, years))
                        if not disease_from_history:
                            disease_from_history = disease
                        break
        
        # 临床推理：根据典型症状+体征+检验数据推断最可能诊断
        # 诊断赋值
        # 急症已在 generate() 中通过 _check_emergency_diagnosis 处理
        # 此处处理：纯数据输入、病史来源、外院诊断、LLM诊断
        
        if input_type == "data":
            has_lab_data = info.blood_sugar or info.blood_sugar_2h or info.hba1c or info.blood_pressure or info.lab_exam_raw
            if not has_lab_data:
                has_lab_data = bool(re.search(r'(空腹|餐后|糖化|mmol|血糖|血压)\D*\d+', raw_input))
            # 纯数据模式：默认需结合临床。仅急诊快检结果可覆盖
            if info.disease_type and "急" in info.disease_type:
                record.preliminary_diagnosis = [info.disease_type]
            elif has_lab_data:
                record.preliminary_diagnosis = ["待补充"]
            else:
                record.preliminary_diagnosis = ["待补充"]
        elif info.disease_type and info.disease_type not in ("", "待补充"):
            # 急诊/LLM诊断，但需要融合病史来源标注
            dt_clean = info.disease_type.strip()
            # 如果LLM诊断是纯疾病名且与病史匹配，使用病史标注版
            history_match = None
            for d, y in disease_history_list:
                if d in dt_clean or dt_clean in d:
                    history_match = (d, y)
                    break
            if history_match and disease_history_list:
                # 病史优先（带来源标注），LLM诊断兜底
                record.preliminary_diagnosis = []
                for d, y in disease_history_list:
                    record.preliminary_diagnosis.append(f"{d}（来源：患者自述，{y}）")
                # 如果LLM有不同于病史的诊断，追加
                for d, y in disease_history_list:
                    if d in dt_clean:
                        dt_clean = dt_clean.replace(d, "").strip("，,、 ")
                if dt_clean and dt_clean not in ("可能", "可能，建议进一步检查"):
                    record.preliminary_diagnosis.append(dt_clean)
            else:
                # 急诊/LLM诊断优先于病史列举
                record.preliminary_diagnosis = [info.disease_type]
                if disease_history_list:
                    for d, y in disease_history_list:
                        if d not in info.disease_type:
                            record.preliminary_diagnosis.append(f"{d}（来源：患者自述，{y}）")
        elif disease_history_list:
            # 病史中的已知疾病优先，标注来源
            diag_items = []
            for d, y in disease_history_list:
                diag_items.append(f"{d}（来源：患者自述，{y}）")
            record.preliminary_diagnosis = diag_items
        elif info.has_diagnosis_source and info.diagnosis_source:
            record.preliminary_diagnosis = [f"{info.diagnosis_source}（来源：患者自述/外院诊断）"]
        elif info.disease_type:
            dt = info.disease_type
            logger.info(f"ADMISSION_NOTE: info.disease_type = {dt!r}")
            if any(kw in dt for kw in ['可能', '建议', '需结合', '待查', '随访', '（', '来源']):
                record.preliminary_diagnosis = [dt]
            elif info.positive_symptoms or info.imaging_exam:
                record.preliminary_diagnosis = [f"{dt}可能，建议进一步检查"]
            else:
                record.preliminary_diagnosis = [f"{dt}可能，建议进一步检查"]







        
        # 治疗计划
        if is_dialogue:
            # 对话格式：从原始文本中提取药物
            dialogue_meds = extract_medications_from_text(raw_input)
            if dialogue_meds:
                record.treatment_plan = "；".join(dialogue_meds)
            elif info.treatment_plan:
                record.treatment_plan = info.treatment_plan
            else:
                record.treatment_plan = cls._generate_treatment_plan(record.preliminary_diagnosis, info.disease_type, info.symptoms, raw_input)
        else:
            record.treatment_plan = info.treatment_plan if info.treatment_plan else cls._generate_treatment_plan(record.preliminary_diagnosis, info.disease_type, info.symptoms, raw_input)
        
        # 对话格式：诊断去除"可能，建议进一步检查"后缀
        if is_dialogue and record.preliminary_diagnosis:
            cleaned = [clean_diagnosis_for_dialogue(d) for d in record.preliminary_diagnosis]
            # 如果清理后为空或只剩"待补充"，保留原始诊断
            valid = [d for d in cleaned if d and d != "待补充"]
            if valid:
                record.preliminary_diagnosis = valid
            # 同时清理 "待补充" → 继续保留系统现有逻辑
        
        # 个人史、婚育史、家族史（从原始输入提取）
        personal_parts = []
        # 月经婚育史
        menses_match = re.search(r'月经[^，。；]*', raw_input)
        if menses_match:
            personal_parts.append(menses_match.group(0))
        # 婚育史
        marriage_match = re.search(r'(已婚|未婚|离异|丧偶)[^，。；]*', raw_input)
        if marriage_match:
            personal_parts.append(marriage_match.group(0))
        
        record.personal_history = "、".join(personal_parts) if personal_parts else "未提供"
        record.marital_history = "未提供"
        record.family_history = "待补充"
        
        # 最终清理：去除所有文本字段中的换行符
        for field_name in ['chief_complaint', 'present_illness', 'past_history', 
                           'personal_history', 'family_history', 'physical_exam',
                           'auxiliary_exam', 'treatment_plan']:
            val = getattr(record, field_name, None)
            if val and isinstance(val, str):
                setattr(record, field_name, ' '.join(val.split()))
        
        return record
    
    @classmethod
    def fill_follow_up_record(cls, info: ExtractedInfo, input_type: str = "", raw_input: str = "") -> MedicalRecord:
        """填充随访记录模板（增强版：含症状变化评估、用药依从性、下次随访建议）"""
        record = MedicalRecord(record_type=MedicalRecordType.FOLLOW_UP_RECORD)
        
        # 复查/复诊场景：优先从已知诊断生成主诉
        if input_type in ["复诊", "随访", "复查"] or "复查" in raw_input or "复诊" in raw_input:
            clean_disease = info.disease_type if info.disease_type else ""
            clean_disease = re.sub(r'[（(][^）)]*[）)]', '', clean_disease).strip()
            clean_disease = re.sub(r'[，,]\s*复查$', '', clean_disease).strip()
            if clean_disease and clean_disease not in ["待补充", "需结合临床表现及重复检测确认"]:
                record.chief_complaint = f"{clean_disease}复查"
            elif info.chief_complaint:
                record.chief_complaint = info.chief_complaint
            else:
                record.chief_complaint = "复查"
        elif info.chief_complaint and len(info.chief_complaint) >= 4 and '待查' not in info.chief_complaint and '未能' not in info.chief_complaint:
            record.chief_complaint = info.chief_complaint
        elif info.disease_type:
            record.chief_complaint = f"{info.disease_type}随访"
        else:
            record.chief_complaint = "待补充"
        
        # 现病史（含症状变化评估）
        present_illness_parts = []
        if info.gender and info.age:
            present_illness_parts.append(f"患者{info.gender}，{info.age}")
        if info.disease_type:
            present_illness_parts.append(f"{info.disease_type}随访")
        
        # 症状变化评估
        if info.symptoms:
            symptom_list = info.symptoms if isinstance(info.symptoms, list) else [info.symptoms]
            present_illness_parts.append(f"本次症状：{'、'.join(symptom_list)}")
        
        # 用药信息
        if info.medications:
            present_illness_parts.append(f"目前用药：{info.medications}")
        
        record.present_illness = cls._build_present_illness_narrative(present_illness_parts)
        
        # 辅助检查
        auxiliary_exam_parts = []
        if info.blood_sugar:
            auxiliary_exam_parts.append(f"空腹血糖：{info.blood_sugar}")
        if info.blood_sugar_2h:
            auxiliary_exam_parts.append(f"餐后2小时血糖：{info.blood_sugar_2h}")
        if info.hba1c:
            auxiliary_exam_parts.append(f"糖化血红蛋白：{info.hba1c}")
        if info.blood_pressure:
            auxiliary_exam_parts.append(f"血压：{info.blood_pressure}")
        if info.heart_rate:
            auxiliary_exam_parts.append(f"心率：{info.heart_rate}")
        if info.blood_oxygen:
            auxiliary_exam_parts.append(f"血氧：{info.blood_oxygen}%")
        if info.tsh:
            auxiliary_exam_parts.append(f"TSH：{info.tsh}")
        if info.ft3:
            auxiliary_exam_parts.append(f"FT3：{info.ft3}")
        if info.ft4:
            auxiliary_exam_parts.append(f"FT4：{info.ft4}")
        if info.cea:
            auxiliary_exam_parts.append(f"CEA：{info.cea}")
        if info.ca199:
            auxiliary_exam_parts.append(f"CA199：{info.ca199}")
        if info.ca153:
            auxiliary_exam_parts.append(f"CA153：{info.ca153}")
        if info.creatinine:
            auxiliary_exam_parts.append(f"肌酐：{info.creatinine}")
        if info.bun:
            auxiliary_exam_parts.append(f"尿素氮：{info.bun}")
        if info.afp:
            auxiliary_exam_parts.append(f"AFP：{info.afp}")
        if info.hbv_dna:
            auxiliary_exam_parts.append(f"HBV-DNA：{info.hbv_dna}")
        if info.pt:
            auxiliary_exam_parts.append(f"PT：{info.pt}s")
        if info.aptt:
            auxiliary_exam_parts.append(f"APTT：{info.aptt}s")
        if info.fib:
            auxiliary_exam_parts.append(f"FIB：{info.fib}")
        if info.d_dimer:
            auxiliary_exam_parts.append(f"D-二聚体：{info.d_dimer}")
        if info.esr:
            auxiliary_exam_parts.append(f"血沉：{info.esr}")
        if info.crp:
            auxiliary_exam_parts.append(f"CRP：{info.crp}")
        if info.alt:
            auxiliary_exam_parts.append(f"ALT：{info.alt}")
        if info.ast:
            auxiliary_exam_parts.append(f"AST：{info.ast}")
        if info.tc:
            auxiliary_exam_parts.append(f"TC：{info.tc}")
        if info.tg:
            auxiliary_exam_parts.append(f"TG：{info.tg}")
        if info.ldl_c:
            auxiliary_exam_parts.append(f"LDL-C：{info.ldl_c}")
        if info.hdl_c:
            auxiliary_exam_parts.append(f"HDL-C：{info.hdl_c}")
        if info.uric_acid:
            auxiliary_exam_parts.append(f"尿酸：{info.uric_acid}")
        
        if auxiliary_exam_parts:
            record.auxiliary_exam = "".join(auxiliary_exam_parts)
        else:
            record.auxiliary_exam = "本次无辅助检查"
        
        # 体格检查（生命体征）
        physical_exam_lines = []
        vitals = extract_vitals_from_text(raw_input) if raw_input else {}
        gender_part = info.gender if info.gender else "待补充"
        age_part = info.age.replace('岁', '') if info.age else "待补充"
        physical_exam_lines.append(f"- 一般情况：{gender_part}，{age_part}岁")
        t = vitals.get('temperature', info.temperature if info.temperature else '')
        bp = vitals.get('bp', info.blood_pressure if info.blood_pressure else '')
        hr = vitals.get('hr', info.heart_rate if info.heart_rate else '')
        rr = vitals.get('rr', info.respiratory_rate if info.respiratory_rate else '')
        vs_items = []
        if t: vs_items.append(f"T{t}℃" if '℃' not in str(t) else str(t))
        if hr: vs_items.append(f"P{hr}次/分" if '次' not in str(hr) else str(hr))
        if rr: vs_items.append(f"R{rr}次/分" if '次' not in str(rr) else str(rr))
        bp_clean = bp.replace('BP', '').replace('bp', '').strip() if bp else bp
        if bp_clean: vs_items.append(f"BP{bp_clean}")
        vs_display = "，".join(vs_items) if vs_items else "待补充"
        physical_exam_lines.append(f"- 生命体征：{vs_display}")
        record.physical_exam = "\n".join(physical_exam_lines)
        
        # 随访评估（用药依从性 + 生活方式）
        assessment_parts = []
        if info.medications:
            assessment_parts.append("用药依从性：规律服药（患者自述）")
        else:
            assessment_parts.append("用药情况：无明确用药记录，需进一步确认")
        
        # 生活方式建议（基于诊断）
        diag_text = (info.disease_type or "") + " ".join(str(d) for d in (record.preliminary_diagnosis or []))
        if "糖尿病" in diag_text:
            assessment_parts.append("饮食控制：低糖饮食，定时定量")
            assessment_parts.append("运动建议：每周至少150分钟中等强度运动")
        if "高血压" in diag_text:
            assessment_parts.append("生活方式：低盐低脂饮食，戒烟限酒")
            assessment_parts.append("家庭血压监测：每日早晚各测一次")
        if "肝炎" in diag_text or "肝功能" in diag_text:
            assessment_parts.append("生活方式：戒酒，避免肝毒性药物")
        if "肾炎" in diag_text or "肾病" in diag_text or "肾功能" in diag_text:
            assessment_parts.append("饮食控制：低盐低蛋白饮食")
        
        record.personal_history = "；".join(assessment_parts) if assessment_parts else "待补充"
        
        # 治疗计划
        if info.medications:
            record.treatment_plan = f"继续目前用药：{info.medications}；定期随访复查"
        else:
            diag_list = record.preliminary_diagnosis or [info.disease_type] if info.disease_type else ["待补充"]
            record.treatment_plan = cls._generate_treatment_plan(diag_list, info.disease_type, info.symptoms if isinstance(info.symptoms, list) else [], raw_input)
        
        # 下次随访建议
        if "糖尿病" in diag_text or "高血压" in diag_text:
            record.family_history = "建议1-3个月后随访复查"
        elif "肝炎" in diag_text:
            record.family_history = "建议3-6个月后随访复查肝功能及病毒指标"
        else:
            record.family_history = "根据病情变化随时复诊"
        
        # 初步诊断
        if info.disease_type:
            record.preliminary_diagnosis = [f"{info.disease_type}（随访）"]
        else:
            record.preliminary_diagnosis = ["待补充"]
        
        # 主诉清理
        if record.chief_complaint:
            record.chief_complaint = ' '.join(record.chief_complaint.split())
        if record.present_illness:
            record.present_illness = ' '.join(record.present_illness.split())
        
        return record

    @classmethod
    def fill_discharge_summary(cls, info: ExtractedInfo, input_type: str = "", raw_input: str = "") -> MedicalRecord:
        """填充出院小结模板"""
        record = MedicalRecord(record_type=MedicalRecordType.DISCHARGE_SUMMARY)
        
        # 主诉（出院时总结）- 过滤fallback值
        FALLBACK_CC_PATTERNS = ['患者提供', '请提供', '待补充', '请描述', '输入过长', '未能识别', '检验数据']
        if info.chief_complaint and len(info.chief_complaint) >= 4 and not any(p in info.chief_complaint for p in FALLBACK_CC_PATTERNS):
            record.chief_complaint = info.chief_complaint
        elif info.symptoms and len(info.symptoms) > 0:
            # 从症状+病史构建主诉
            key_symptoms = [s for s in info.symptoms if s not in ['发热']][:3]
            if key_symptoms:
                record.chief_complaint = f"{'、'.join(key_symptoms)}"
                if info.medical_history:
                    history_hint = info.medical_history[0][:8]
                    record.chief_complaint = f"{history_hint}，{record.chief_complaint}"
        elif info.disease_type and info.disease_type != "待补充":
            record.chief_complaint = f"{info.disease_type}出院"
        else:
            record.chief_complaint = "待补充"
        
        # 入院情况（过滤CC的fallback值）
        admission_parts = []
        if info.gender and info.age:
            admission_parts.append(f"患者{info.gender}，{info.age}")
        admission_cc = info.chief_complaint
        if admission_cc and not any(p in admission_cc for p in FALLBACK_CC_PATTERNS):
            admission_parts.append(f'因"{admission_cc}"入院')
        elif info.symptoms:
            admission_parts.append(f'因"{",".join(info.symptoms[:3])}"入院')
        elif info.disease_type and info.disease_type != "待补充":
            admission_parts.append(f'因"{info.disease_type}"入院')
        if info.symptoms:
            admission_parts.append(f"主要症状：{'、'.join(info.symptoms)}")
        if info.medical_history:
            admission_parts.append(f"既往史：{'、'.join(info.medical_history)}")
        record.present_illness = cls._build_present_illness_narrative(admission_parts)
        
        # 诊疗经过（从输入提取治疗相关信息）
        treatment_parts = []
        if info.physical_exam_raw:
            treatment_parts.append(f"查体：{info.physical_exam_raw}")
        if info.lab_exam_raw:
            treatment_parts.append(f"辅助检查：{info.lab_exam_raw}")
        if info.imaging_exam:
            treatment_parts.append(f"影像学：{'；'.join(info.imaging_exam)}")
        # 提取手术/操作
        surgery_match = re.search(r'(行|接受|完成|予)([一-龥A-Za-z0-9]+?(?:术|治疗|检查|植入|置入|切开|切除|引流|清创|换药|透析))', raw_input)
        if surgery_match:
            treatment_parts.append(f"主要治疗：{surgery_match.group(0)}")
        record.past_history = cls._build_present_illness_narrative(treatment_parts) if treatment_parts else "详见入院记录"
        
        # 辅助检查（出院时关键指标）
        auxiliary_exam_parts = []
        if info.blood_sugar:
            auxiliary_exam_parts.append(f"血糖：{info.blood_sugar}")
        if info.blood_pressure:
            auxiliary_exam_parts.append(f"血压：{info.blood_pressure}")
        if info.alt:
            auxiliary_exam_parts.append(f"ALT：{info.alt}")
        if info.ast:
            auxiliary_exam_parts.append(f"AST：{info.ast}")
        if info.creatinine:
            auxiliary_exam_parts.append(f"肌酐：{info.creatinine}")
        if info.wbc:
            auxiliary_exam_parts.append(f"WBC：{info.wbc}×10^9/L")
        if info.hb:
            auxiliary_exam_parts.append(f"HGB：{info.hb}")
        if info.crp:
            auxiliary_exam_parts.append(f"CRP：{info.crp}")
        if auxiliary_exam_parts:
            record.auxiliary_exam = "".join(auxiliary_exam_parts)
        else:
            record.auxiliary_exam = "本次无辅助检查"
        
        # 体格检查
        physical_exam_lines = []
        vitals = extract_vitals_from_text(raw_input) if raw_input else {}
        gender_part = info.gender if info.gender else "待补充"
        age_part = info.age.replace('岁', '') if info.age else "待补充"
        physical_exam_lines.append(f"- 一般情况：{gender_part}，{age_part}岁")
        t = vitals.get('temperature', info.temperature if info.temperature else '')
        bp = vitals.get('bp', info.blood_pressure if info.blood_pressure else '')
        hr = vitals.get('hr', info.heart_rate if info.heart_rate else '')
        vs_items = []
        if t: vs_items.append(f"T{t}℃" if '℃' not in str(t) else str(t))
        if hr: vs_items.append(f"P{hr}次/分" if '次' not in str(hr) else str(hr))
        bp_clean = bp.replace('BP', '').replace('bp', '').strip() if bp else bp
        if bp_clean: vs_items.append(f"BP{bp_clean}")
        vs_display = "，".join(vs_items) if vs_items else "待补充"
        physical_exam_lines.append(f"- 生命体征：{vs_display}")
        record.physical_exam = "\n".join(physical_exam_lines)
        
        # 出院诊断
        if info.disease_type and info.disease_type != "待补充":
            record.preliminary_diagnosis = [f"{info.disease_type}（出院诊断）"]
        elif info.medical_history:
            diag_items = []
            for h in info.medical_history:
                diag_items.append(f"{h}（出院诊断）")
            record.preliminary_diagnosis = diag_items
        else:
            record.preliminary_diagnosis = ["待补充"]
        
        # 出院医嘱（从输入中提取用药+建议）
        if info.medications and '待补充' not in info.medications:
            record.treatment_plan = f"出院带药：{info.medications}"
        elif info.treatment_plan and '待补充' not in info.treatment_plan:
            record.treatment_plan = f"出院医嘱：{info.treatment_plan}"
        else:
            # 尝试从输入中提取用药相关内容
            med_match = re.search(r'(?:出院带药|继续口服|继续服用|出院后|出院.*?(?:口服|服用|治疗|用药))[：:]*([一-龥A-Za-z0-9\s，,、.。/；;（）()]+?)(?:[。；;]|$)', raw_input)
            if med_match:
                record.treatment_plan = f"出院带药：{med_match.group(1).strip()}"
            # 再尝试提取输入中明确的药物名+用法
            elif re.search(r'(?:阿司匹林|氯吡格雷|二甲双胍|胰岛素|头孢|阿莫西林|硝苯地平|氨氯地平|缬沙坦|厄贝沙坦|美托洛尔|辛伐他汀|阿托伐他汀|华法林|利伐沙班|达比加群|奥美拉唑|泮托拉唑|甲硝唑|左氧氟沙星|莫西沙星|沙丁胺醇|噻托溴铵|布地奈德)', raw_input):
                med_full = re.search(r'((?:阿司匹林|氯吡格雷|二甲双胍|胰岛素|头孢\w*|阿莫西林|硝苯地平|氨氯地平|缬沙坦|厄贝沙坦|美托洛尔|辛伐他汀|阿托伐他汀|华法林|利伐沙班|达比加群|奥美拉唑|泮托拉唑|甲硝唑|左氧氟沙星|莫西沙星|沙丁胺醇|噻托溴铵|布地奈德)\s*[一-鿿A-Za-z0-9\s，,.、（）()，/\d]+)', raw_input)
                if med_full:
                    record.treatment_plan = f"出院带药：{med_full.group(1).strip()}"
                else:
                    record.treatment_plan = "出院医嘱：继续药物治疗，定期门诊随访"
            else:
                diag_list = record.preliminary_diagnosis or [info.disease_type] if info.disease_type else ["待补充"]
                record.treatment_plan = cls._generate_treatment_plan(diag_list, info.disease_type, info.symptoms if isinstance(info.symptoms, list) else [], raw_input)
        
        # 出院情况（从输入中提取恢复/改善描述，排除"出院带药"等非恢复描述）
        improvement_match = re.search(r'(?:恢复|缓解|好转|改善|消失|稳定|愈合|减轻)[^。，；;]*[好善退解复常停]?', raw_input)
        if improvement_match:
            result_text = improvement_match.group(0)
            # 过滤：不含"带药""继续""建议"等非恢复描述
            if not re.search(r'(?:带药|口服|服用|建议|嘱)', result_text):
                record.family_history = f"出院情况：{result_text}"
            else:
                record.family_history = "出院情况：待补充"
        else:
            record.family_history = "出院情况：待补充"
        
        # 随访建议
        record.personal_history = "建议定期门诊随访，如有不适随时就诊"
        
        # 清理所有字段
        for field_name in ['chief_complaint', 'present_illness', 'past_history', 
                           'personal_history', 'family_history', 'physical_exam',
                           'auxiliary_exam', 'treatment_plan']:
            val = getattr(record, field_name, None)
            if val and isinstance(val, str):
                setattr(record, field_name, ' '.join(val.split()))
        
        return record


# ====== 对话格式 vs 简单文本格式检测与处理 ======

def detect_dialogue_format(text: str) -> bool:
    """检测输入是否为医患对话格式（含"患者："/"医生："标记）"""
    text_clean = text.replace('', ' ')
    has_patient = bool(re.search(r'(?:患者|家属)[\s：:]\s*', text_clean))
    has_doctor = bool(re.search(r'(?:医生|大夫)[\s：:]\s*', text_clean))
    return has_patient and has_doctor


def extract_vitals_from_text(text: str) -> Dict[str, str]:
    """从原始文本中提取生命体征"""
    result = {}
    # 体温：优先 "体温37.8℃" 或 "体温：37.8" 或 "最高39.0℃"
    # 注意：避免将 ALT/AST/Cr 等检验值误判为体温
    m = re.search(r'(?:体温|T)[：:]*\s*(\d+(?:\.\d+)?)\s*[℃°]?', text)
    if m:
        temp_val = float(m.group(1))
        if 34 <= temp_val <= 43:  # 合理体温范围
            result['temperature'] = m.group(1)
    if not result.get('temperature'):
        m = re.search(r'最高(\d+(?:\.\d+)?)\s*[℃°]', text)
        if m:
            temp_val = float(m.group(1))
            if 34 <= temp_val <= 43:
                result['temperature'] = m.group(1)
    if not result.get('temperature'):
        # 口语"39度" - 但必须前面有体温/发热/烧等上下文
        m = re.search(r'(?:[体发]|体温|发烧|发热).*?(\d+(?:\.\d+)?)\s*度', text)
        if m:
            temp_val = float(m.group(1))
            if 34 <= temp_val <= 43:
                result['temperature'] = temp_val if '.' in str(temp_val) else f"{temp_val}.0"

    # 血压
    m = re.search(r'(?:BP|血压)[：:]*\s*(\d+)/(\d+)', text)
    if m: result['bp'] = f"{m.group(1)}/{m.group(2)}"
    else:
        m = re.search(r'(\d{2,3})/(\d{2,3})\s*(?:mmHg)?', text)
        if not m:
            m = re.search(r'血压\s*(\d{2,3})[^\d]*(\d{2,3})', text)
        if m and 50 <= int(m.group(1)) <= 260 and 30 <= int(m.group(2)) <= 200:
            result['bp'] = f"{m.group(1)}/{m.group(2)}"
    # 心率/脉搏
    m = re.search(r'(?:HR|心率|(?<![a-zA-Z])P|脉搏)[：:]*\s*(\d+)', text)
    if m: result['hr'] = m.group(1)
    else:
        m = re.search(r'心率\s*(\d+)', text)
        if m: result['hr'] = m.group(1)
    # 呼吸
    m = re.search(r'(?:RR|R|呼吸)[：:]*\s*(\d+)', text)
    if m: result['rr'] = m.group(1)
    # 血氧
    m = re.search(r'(?:SpO2|血氧)[：:]*\s*(\d+)', text)
    if m: result['spo2'] = m.group(1)
    return result


def extract_medications_from_text(text: str) -> List[str]:
    """从医患对话文本中提取药物及治疗措施"""
    meds = []
    patterns = [
        r'(?:阿司匹林|氯吡格雷|阿托伐他汀|硝酸甘油|布洛芬|对乙酰氨基酚|'
        r'头孢呋辛|头孢曲松|头孢他啶|头孢克肟|阿莫西林|'
        r'甲硝唑|奥硝唑|替硝唑|莫西沙星|左氧氟沙星|青霉素|氨苄西林|'
        r'罗红霉素|克拉霉素|阿奇霉素|'
        r'二甲双胍|胰岛素|格列|阿卡波糖|'
        r'氨氯地平|硝苯地平|卡托普利|缬沙坦|厄贝沙坦|氯沙坦|'
        r'美托洛尔|比索洛尔|呋塞米|螺内酯|氢氯噻嗪|'
        r'奥美拉唑|泮托拉唑|雷贝拉唑|多潘立酮|莫沙必利|'
        r'叶酸片|叶酸|维生素B|维生素C|维生素D|钙片|骨化三醇|'
        r'泼尼松|地塞米松|甲泼尼龙|'
        r'布地奈德|沙丁胺醇|异丙托溴铵|噻托溴铵|'
        r'甘露醇|破伤风|狂犬疫苗|'
        r'口服补液盐|蒙脱石散|双歧杆菌|蓝芩口服液|'
        r'地屈孕酮|黄体酮|左甲状腺素钠|优甲乐|硒酵母|'
        r'硝苯地平控释片|非洛地平|'
        r'氯化钾|碳酸钙|阿法骨化醇|'
        r'葡醛内酯|多烯磷脂酰胆碱|水飞蓟宾)[^。，、；]*',]

    for pat in patterns:
        for m in re.finditer(pat, text):
            med = m.group(0).strip()
            if med and len(med) >= 3:
                meds.append(med)
    # 去重
    seen = set()
    unique = []
    for m in meds:
        key = m[:6]
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def clean_diagnosis_for_dialogue(diagnosis: str) -> str:
    """对话格式：去除规则引擎的'可能，建议进一步检查'等安全后缀"""
    if not diagnosis:
        return diagnosis
    cleaned = re.sub(r'可能，建议进一步检查$', '', diagnosis)
    cleaned = re.sub(r'可能$', '', cleaned)
    cleaned = re.sub(r'，建议进一步检查$', '', cleaned)
    cleaned = re.sub(r'（.*?）', '', cleaned)
    return cleaned.strip()


def extract_labs_from_dialogue(text: str) -> List[str]:
    """从对话文本中提取检验结果"""
    lab_items = []
    patterns = [
        (r'(?:WBC|白细胞)\s*[:：]?\s*([\d.]+)\s*[×xX*]\s*10\^?9\s*[//]?\s*L', 'WBC×10^9/L'),
        (r'(?:WBC|白细胞)\s*[:：]?\s*([\d.]+)', '白细胞'),
        (r'(?:Hb|血红蛋白)\s*[:：]?\s*(\d+)', 'Hb'),
        (r'(?:PLT|血小板)\s*[:：]?\s*(\d+)', 'PLT'),
        (r'(?:CRP|C反应蛋白|hs-CRP)\s*[:：]?\s*([\d.]+)', 'CRP'),
        (r'(?:PCT|降钙素原)\s*[:：]?\s*([\d.]+)', 'PCT'),
        (r'cTnI|肌钙蛋白\s*[:：]?\s*([\d.]+)', 'cTnI'),
        (r'CK-MB\s*[:：]?\s*([\d.]+)', 'CK-MB'),
        (r'(?:BNP|NT-proBNP|脑钠肽)\s*[:：]?\s*([\d.]+)', 'BNP'),
        (r'空腹血糖\s*[:：]?\s*([\d.]+)', '空腹血糖'),
        (r'HbA1c|糖化血红蛋白\s*[:：]?\s*([\d.]+)', 'HbA1c'),
        (r'(?:ALT|谷丙转氨酶)\s*[:：]?\s*(\d+)', 'ALT'),
        (r'(?:AST|谷草转氨酶)\s*[:：]?\s*(\d+)', 'AST'),
        (r'(?:Cr|肌酐|CREA)\s*[:：]?\s*(\d+)', 'Cr'),
        (r'(?:UA|尿酸)\s*[:：]?\s*(\d+)', 'UA'),
        (r'(?:TC|总胆固醇)\s*[:：]?\s*([\d.]+)', 'TC'),
        (r'(?:TG|甘油三酯)\s*[:：]?\s*([\d.]+)', 'TG'),
        (r'(?:LDL-C|低密度脂蛋白)\s*[:：]?\s*([\d.]+)', 'LDL-C'),
        (r'(?:HDL-C|高密度脂蛋白)\s*[:：]?\s*([\d.]+)', 'HDL-C'),
        (r'(?:K|钾|Na|钠|Cl|氯|Ca|钙)\s*[:：]?\s*([\d.]+)', '电解质'),
        (r'尿HCG\s*[:：]?\s*(阳性|阴性|\+)', '尿HCG'),
        (r'孕酮\s*[:：]?\s*([\d.]+)', '孕酮'),
        (r'大便常规.*?WBC\s*(\d+)\s*[-~]\s*(\d+)', '大便常规'),
        (r'中性粒.*?(\d+)%', '中性粒细胞'),
        (r'血型\s*[:：]?\s*([A-Z]+型)', '血型'),
        (r'PT\s*[:：]?\s*([\d.]+)', 'PT'),
        (r'APTT\s*[:：]?\s*([\d.]+)', 'APTT'),]

    for pat, label in patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(0).strip()
            lab_items.append(val)
    return lab_items



def _check_emergency_diagnosis(text: str, info: 'ExtractedInfo') -> str:
    """急症快速诊断检查（仅10个高致命性急症，基于明确体征/检验组合）
    返回诊断字符串或空字符串（非急症）
    """
    all_text = (info.physical_exam_raw or "") + (info.lab_exam_raw or "") + text
    
    # 1. 急性阑尾炎：右下腹痛+反跳痛+白细胞升高
    if any(kw in all_text for kw in ['右下腹', '麦氏点']) and any(kw in all_text for kw in ['反跳痛', '压痛']):
        if any(kw in all_text for kw in ['WBC', '白细胞', '中性粒']):
            return "急性阑尾炎可能，建议进一步检查"
    
    # 2. 急性冠脉综合征：胸痛+心电图异常或心肌酶升高
    if any(kw in all_text for kw in ['胸痛', '胸闷']) and any(kw in all_text for kw in ['压榨', '放射', '大汗']):
        if any(kw in all_text for kw in ['心电图', 'ST段', 'T波', '心肌酶', '肌钙蛋白', 'BNP']):
            return "急性冠脉综合征可能，建议紧急检查心肌标志物"
    
    # 3. 蛛网膜下腔出血：突发剧烈头痛+颈抵抗/呕吐
    if any(kw in all_text for kw in ['剧烈头痛', '突发头痛', '爆炸样头痛']):
        if any(kw in all_text for kw in ['颈抵抗', '呕吐', '意识障碍', '昏迷']):
            return "蛛网膜下腔出血可能，建议紧急头颅CT检查"
    
    # 4. 创伤性休克：外伤+意识障碍+低血压
    if any(kw in all_text for kw in ['车祸', '外伤', '撞击', '坠落', '跌倒']):
        if any(kw in all_text for kw in ['意识模糊', '昏迷', '意识障碍', '休克']):
            if any(kw in all_text for kw in ['血压']) and re.search(r'血压\D*\d{2,3}\D*/\D*\d{2,3}', all_text):
                bp_match = re.search(r'血压\D*(\d{2,3})\D*/\D*(\d{2,3})', all_text)
                if bp_match and int(bp_match.group(1)) < 90:
                    return "创伤性休克可能，多发伤可能，建议紧急抢救及进一步检查"
            return "多发伤可能，建议急诊处理"
    
    # 5. 肺栓塞：突发呼吸困难+胸痛+D-二聚体升高
    if any(kw in all_text for kw in ['突发呼吸困难', '胸痛', '咯血']):
        if any(kw in all_text for kw in ['D-二聚体', 'D二聚体', 'd-dimer', '晕厥']):
            return "肺栓塞可能，建议紧急CTPA检查"
    
    # 6. 脑出血/脑梗死：突发偏瘫+意识障碍+高血压史
    if any(kw in all_text for kw in ['偏瘫', '肢体无力', '口角歪斜', '言语不清', '失语']):
        if any(kw in all_text for kw in ['突发', '突然']) or any(kw in info.chief_complaint or '' for kw in ['突发', '突然']):
            return "急性脑血管病可能（脑梗死/脑出血），建议紧急头颅CT检查"
    
    # 7. 急性胰腺炎：上腹痛+淀粉酶升高
    if any(kw in all_text for kw in ['上腹痛', '上腹部', '左上腹']) and any(kw in all_text for kw in ['淀粉酶', '胰淀粉酶']):
        return "急性胰腺炎可能，建议进一步检查"
    
    # 8. 主动脉夹层：撕裂样胸背痛+血压显著升高
    if any(kw in all_text for kw in ['撕裂样', '刀割样']) and any(kw in all_text for kw in ['胸痛', '背痛', '胸背痛']):
        return "主动脉夹层可能，建议紧急血管CTA检查"
    
    # 9. 宫外孕破裂：停经+腹痛+休克
    if any(kw in all_text for kw in ['停经', '月经推迟']):
        if any(kw in all_text for kw in ['下腹痛', '腹痛']) and any(kw in all_text for kw in ['休克', '血压下降', '昏厥', '晕倒']):
            return "异位妊娠破裂可能，建议紧急妇科会诊"
    
    # 10. 心力衰竭急性发作：胸闷+BNP升高+下肢水肿
    if any(kw in all_text for kw in ['胸闷', '呼吸困难', '喘憋']) and any(kw in all_text for kw in ['BNP', 'NT-proBNP', 'proBNP']):
        diag = "心力衰竭可能"
        # 检测房颤
        if any(kw in all_text for kw in ['房颤', '心房颤动', '心房纤颤', 'Afib', 'AF']):
            diag += "，心房颤动"
        diag += "，建议进一步检查"
        return diag
    
    # 11. 心力衰竭风险筛查：胸闷加重+高龄+心血管危险因素
    if any(kw in all_text for kw in ['胸闷', '呼吸困难']) and any(kw in all_text for kw in ['加重', '持续', '越来越重', '更重']):
        age_match = re.search(r'(\d{2})岁', all_text)
        if age_match and int(age_match.group(1)) >= 60:
            if any(kw in all_text for kw in ['高血压', '糖尿病', '冠心病', '心衰', '心力衰竭']):
                return "心力衰竭可能，建议进一步检查BNP及心脏超声"
    
    return ""  # 非急症，交给LLM



def _extract_data_cc(text: str) -> str:
    """从数据主导型输入中提取语义主诉"""
    # 体检场景
    if re.search(r'(?:体检|健康检查|例行检查)', text):
        # 有具体发现
        findings = re.findall(r'(?:发现|示|提示|见)(.{2,10}?(?:结节|占位|肿块|脂肪肝|囊肿|息肉|异常|升高|偏高|降低))', text)
        if findings:
            key_finding = findings[0].strip('，。、 ')
            return f"体检发现{key_finding}复查"
        return "体检报告解读"
    # 复查/随访场景
    if re.search(r'(?:复查|复诊|随访|术后)', text):
        diseases = re.findall(r'(?:高血压|糖尿病|冠心病|慢阻肺|乙肝|甲亢|类风湿|肾病|肿瘤|癌)', text)
        if diseases:
            return f"{'、'.join(diseases[:2])}复查"
        return "复查"
    # 检验数据提供
    if re.search(r'(?:检验|化验|报告|结果)', text):
        return "检验报告解读"
    return ""  # 返回空则使用默认占位符

def _check_common_diagnosis(text: str, info) -> str:
    """纯规则常见病诊断引擎（40种基层常见病，零API成本）
    用作API版降级方案或独立诊断引擎。
    返回诊断字符串或空字符串。
    """
    all_text = (info.physical_exam_raw or "") + (info.lab_exam_raw or "") + text
    cc = info.chief_complaint or ""
    symptoms = (info.positive_symptoms or []) + (info.negative_symptoms or [])
    symptoms_str = " ".join(symptoms)
    
    # ================================================================
    # 呼吸系统 (Respiratory)
    # ================================================================
    
    # 1. 社区获得性肺炎：咳嗽+发热+胸片异常 或 咳嗽+发热+啰音
    if any(k in all_text for k in ["咳嗽", "咳"]):
        if any(k in all_text for k in ["发热", "发烧", "体温"]):
            if any(k in all_text for k in ["啰音", "胸片", "CT", "斑片", "实变", "浸润"]):
                return "肺炎可能，建议进一步检查"
    
    # 2. 急性支气管炎：咳嗽+发热+无胸片异常
    if any(k in all_text for k in ["咳嗽", "咳"]) and any(k in all_text for k in ["发热", "发烧"]):
        if not any(k in all_text for k in ["啰音", "斑片", "实变"]):
            return "急性支气管炎可能"
    
    # 3. 上呼吸道感染：咽痛/鼻塞+发热+无咳痰
    if any(k in all_text for k in ["咽痛", "咽喉痛", "嗓子疼", "喉咙痛", "鼻塞", "流涕", "喷嚏"]):
        if any(k in all_text for k in ["发热", "发烧"]):
            return "急性上呼吸道感染可能"
    
    # 4. 支气管哮喘：喘息+哮鸣音
    if any(k in all_text for k in ["喘息", "哮鸣音", "支气管哮喘", "哮喘"]) and any(k in all_text for k in ["呼吸困难", "喘", "气促"]):
        return "支气管哮喘发作可能"
    
    # ================================================================
    # 心血管系统 (Cardiovascular)
    # ================================================================
    
    # 5. 高血压
    bp_match = re.search(r"(?:血压|BP)[^\d]*(\d{2,3})\s*/\s*(\d{2,3})", all_text)
    if bp_match:
        sbp, dbp = int(bp_match.group(1)), int(bp_match.group(2))
        if sbp >= 140 or dbp >= 90:
            grade = "1级" if sbp < 160 else ("2级" if sbp < 180 else "3级")
            return f"高血压病{grade}"
    
    # 6. 心力衰竭
    if any(k in all_text for k in ["呼吸困难", "喘憋", "端坐"]) and any(k in all_text for k in ["下肢水肿", "水肿", "浮肿"]):
        if any(k in all_text for k in ["心脏扩大", "BNP", "proBNP", "NT-proBNP"]):
            return "心力衰竭可能，建议进一步检查"
    
    # ================================================================
    # 消化系统 (Digestive)
    # ================================================================
    
    # 7. 急性胃肠炎：腹泻+呕吐 或 腹泻+腹痛
    if any(k in all_text for k in ["腹泻", "拉肚子", "稀便", "水样便"]):
        if any(k in all_text for k in ["呕吐", "恶心"]) or any(k in all_text for k in ["腹痛", "肚子疼"]):
            return "急性胃肠炎可能"
    
    # 8. 消化性溃疡：上腹痛+周期性+空腹痛
    if any(k in all_text for k in ["上腹痛", "上腹部痛", "胃痛", "胃疼"]):
        if any(k in all_text for k in ["空腹", "夜间", "饥饿", "进食缓解"]):
            return "消化性溃疡可能"
    
    # 9. 胃食管反流：烧心+反酸
    if any(k in all_text for k in ["烧心", "反酸", "胸部烧灼", "胃酸"]):
        if any(k in all_text for k in ["胸骨后", "躺下", "进食后", "饱餐"]):
            return "胃食管反流病可能"
    
    # 10. 功能性消化不良：腹胀+打嗝+无器质性异常
    if any(k in all_text for k in ["腹胀", "嗳气", "打嗝", "早饱"]):
        if not any(k in all_text for k in ["溃疡", "糜烂", "占位", "肿瘤"]):
            return "功能性消化不良可能"
    
    # 11. 便秘
    if any(k in all_text for k in ["便秘", "大便干", "排便困难", "几天一次"]) and any(k in all_text for k in ["便血", "肛裂", "肛门"]):
        return "便秘，肛裂可能"
    if any(k in all_text for k in ["便秘", "大便干", "排便困难", "几天一次"]):
        return "便秘"
    
    # ================================================================
    # 内分泌代谢 (Endocrine/Metabolic)
    # ================================================================
    
    # 12. 2型糖尿病：多饮多尿+血糖升高
    if any(k in all_text for k in ["多饮", "多尿", "口渴"]) and any(k in all_text for k in ["血糖", "HbA1c", "糖化"]):
        return "2型糖尿病可能，建议进一步检查"
    # 单独血糖升高
    fbg_match = re.search(r"(?:空腹血糖|血糖)[^\d]*(\d+\.?\d*)", all_text)
    if fbg_match and float(fbg_match.group(1)) >= 7.0:
        return "2型糖尿病可能，建议进一步检查"
    
    # 13. 糖尿病周围神经病变：糖尿病+肢体麻木
    if any(k in all_text for k in ["糖尿病", "DM", "HbA1c", "糖化"]) and any(k in all_text for k in ["肢体麻木", "手脚麻木", "针刺感", "袜套样"]):
        return "糖尿病周围神经病变可能"
    
    # 14. 甲亢：心悸+手抖+消瘦+怕热
    if sum(1 for k in ["心悸", "手抖", "消瘦", "怕热", "多汗", "甲状腺肿大"] if k in all_text) >= 3:
        return "甲状腺功能亢进症可能，建议查甲功"
    
    # 15. 甲状腺结节：颈部包块+超声
    if any(k in all_text for k in ["颈部包块", "甲状腺结节", "颈前肿物"]):
        return "甲状腺结节，建议超声评估"
    
    # ================================================================
    # 神经系统 (Neurological)
    # ================================================================
    
    # 16. 偏头痛：单侧头痛+畏光/畏声/恶心
    if any(k in all_text for k in ["头痛", "头疼"]) and any(k in all_text for k in ["单侧", "一侧", "搏动"]):
        if any(k in all_text for k in ["恶心", "呕吐", "畏光", "怕光"]):
            return "偏头痛可能"
    
    # 17. 良性位置性眩晕：眩晕+体位改变诱发+无听力下降
    if any(k in all_text for k in ["眩晕", "天旋地转", "头晕"]) and any(k in all_text for k in ["体位", "转头", "翻身", "起床", "躺下"]):
        if "听力" not in all_text:
            return "良性阵发性位置性眩晕可能"
    
    # 18. 特发性震颤：双手震颤+动作时加重+无帕金森表现
    if any(k in all_text for k in ["手抖", "震颤", "手颤"]) and any(k in all_text for k in ["持物", "端杯", "写字", "夹菜", "动作"]):
        if "帕金森" not in all_text and "僵" not in all_text:
            return "特发性震颤可能"
    
    # 19. 帕金森病（早期）：震颤+动作迟缓+肌张力增高
    if any(k in all_text for k in ["手抖", "震颤"]) and any(k in all_text for k in ["动作慢", "走路慢", "小碎步", "僵", "面具脸"]):
        return "帕金森病可能（早期），建议神经内科评估"
    
    # ================================================================
    # 运动系统 (Musculoskeletal)
    # ================================================================
    
    # 20. 腰肌劳损/腰椎间盘突出：腰痛+久坐加重+无外伤
    if any(k in all_text for k in ["腰痛", "腰疼", "腰部不适"]):
        if any(k in all_text for k in ["久坐", "弯腰", "劳累", "搬重"]):
            if not any(k in all_text for k in ["外伤", "车祸", "跌倒"]):
                if any(k in all_text for k in ["下肢放射", "腿麻", "坐骨神经"]):
                    return "腰椎间盘突出症可能"
                return "腰肌劳损可能"
    
    # 21. 颈椎病：颈痛+上肢放射痛/麻木
    if any(k in all_text for k in ["颈痛", "颈部不适", "颈椎"]) and any(k in all_text for k in ["上肢麻木", "手麻", "手臂麻木", "肩背"]):
        return "颈椎病可能"
    
    # 22. 骨折：外伤+畸形/活动受限
    if any(k in all_text for k in ["外伤", "车祸", "跌倒", "摔伤", "扭伤"]):
        if any(k in all_text for k in ["畸形", "不能动", "活动受限", "肿", "不能站立", "不能行走"]):
            body_part = ""
            for bp in ["小腿", "大腿", "前臂", "上臂", "踝", "腕", "膝", "肘"]:
                if bp in all_text:
                    body_part = bp
                    break
            return f"{body_part}骨折可能" if body_part else "骨折可能，建议影像学检查"
    
    # 23. 膝关节炎：膝关节痛+活动时加重+中老年
    if any(k in all_text for k in ["膝盖", "膝关", "膝关节"]):
        age_match = re.search(r"(\d{2})岁", all_text)
        if age_match and int(age_match.group(1)) >= 45:
            return "膝关节骨性关节炎可能"
    
    # 24. 痛风：关节痛+红肿+高尿酸
    if any(k in all_text for k in ["关节红肿", "关节痛"]):
        if any(k in all_text for k in ["尿酸", "痛风", "大脚趾", "第一跖趾"]):
            return "痛风性关节炎可能"
    
    # ================================================================
    # 泌尿生殖 (Urogenital)
    # ================================================================
    
    # 25. 泌尿系感染：尿频+尿急+尿痛
    uti_score = sum(1 for k in ["尿频", "尿急", "尿痛", "排尿痛", "小便痛"] if k in all_text)
    if uti_score >= 2:
        return "泌尿系感染可能"
    
    # 26. 肾结石：腰痛+血尿
    if any(k in all_text for k in ["腰痛", "腰疼"]) and any(k in all_text for k in ["血尿", "尿血", "尿中带血"]):
        return "肾结石可能，建议进一步检查"
    
    # 27. 前列腺增生：老年男性+排尿困难+夜尿增多
    if any(k in all_text for k in ["排尿困难", "夜尿增多", "尿等待", "尿线细", "尿不尽"]):
        gender = info.gender or ""
        age_match = re.search(r"(\d{2})岁", all_text)
        if "男" in gender or (age_match and int(age_match.group(1)) >= 50):
            return "良性前列腺增生可能"
    
    # ================================================================
    # 皮肤 (Dermatological)
    # ================================================================
    
    # 28. 荨麻疹/过敏性皮疹：皮疹+瘙痒
    if any(k in all_text for k in ["皮疹", "荨麻疹", "风团", "红斑"]) and any(k in all_text for k in ["瘙痒", "痒"]):
        return "荨麻疹可能"
    
    # 29. 湿疹：皮肤瘙痒+干燥+脱屑
    if any(k in all_text for k in ["瘙痒", "痒"]) and any(k in all_text for k in ["干燥", "脱屑", "粗糙"]):
        return "湿疹可能"
    
    # 30. 足癣：趾间瘙痒+脱皮
    if any(k in all_text for k in ["脚气", "趾间", "脚趾"]) and any(k in all_text for k in ["痒", "脱皮", "糜烂"]):
        return "足癣可能"
    
    # ================================================================
    # 五官 (ENT/Ophthalmic)
    # ================================================================
    
    # 31. 中耳炎：耳痛+听力下降 或 耳闷+上感
    if any(k in all_text for k in ["耳痛", "耳朵痛"]) and any(k in all_text for k in ["听力下降", "耳闷", "耳鸣"]):
        return "中耳炎可能"
    if any(k in all_text for k in ["耳闷", "耳朵闷", "鼓膜充血"]) and any(k in all_text for k in ["感冒", "上感", "鼻塞"]):
        return "分泌性中耳炎可能"
    
    # 32. 过敏性鼻炎：鼻塞+喷嚏+流涕+季节性
    if any(k in all_text for k in ["鼻塞", "喷嚏"]) and any(k in all_text for k in ["流清涕", "鼻痒", "季节性", "花粉", "过敏"]):
        return "过敏性鼻炎可能"
    
    # 33. 结膜炎：眼红+分泌物
    if any(k in all_text for k in ["眼红", "眼痒", "结膜充血"]) and any(k in all_text for k in ["分泌物", "流泪", "眼屎"]):
        return "结膜炎可能"
    
    # ================================================================
    # 精神心理/其他 (Psychiatric/Other)
    # ================================================================
    
    # 34. 失眠
    if any(k in all_text for k in ["失眠", "入睡困难", "睡眠差", "早醒", "睡不好"]) and any(k in all_text for k in ["疲劳", "乏力", "没精神", "白天困"]):
        return "失眠障碍可能"
    
    # 35. 贫血：面色苍白+乏力+血红蛋白低
    if any(k in all_text for k in ["面色苍白", "乏力", "疲劳"]) and any(k in all_text for k in ["Hb", "血红蛋白", "贫血"]):
        return "贫血待查，建议进一步检查"
    
    # 36. 发热待查
    if any(k in all_text for k in ["发热", "发烧"]):
        temp_match = re.search(r"(?:体温|T\s*|温度)[^\d]*(\d{2}\.?\d*)", all_text)
        if temp_match and float(temp_match.group(1)) >= 38.5:
            if not any(k in all_text for k in ["肺炎", "感染", "炎症"]) and "感染" not in all_text:
                return "发热待查，建议进一步检查"
    
    # 37. 肝功能异常
    if any(k in all_text for k in ["ALT", "AST", "转氨酶"]) and any(k in all_text for k in ["升高", "增高"]):
        if "乙肝" in all_text or "HBsAg" in all_text:
            return "慢性乙型肝炎可能"
        return "肝功能异常待查"
    
    # 38. 高脂血症
    if any(k in all_text for k in ["胆固醇", "甘油三酯", "LDL", "HDL", "血脂"]) and any(k in all_text for k in ["升高", "增高", "偏高"]):
        return "高脂血症"
    
    # 39. 泌尿系结石（无腰痛但B超发现）
    if any(k in all_text for k in ["肾结石", "肾脏结石", "输尿管结石"]) and "B超" in all_text:
        return "肾结石，建议随访"
    
    # 40. 慢性胃炎
    if any(k in all_text for k in ["胃镜", "慢性胃炎", "浅表性胃炎"]):
        if any(k in all_text for k in ["腹胀", "嗳气", "上腹不适", "消化不良"]):
            return "慢性胃炎"
    
    return ""  # 未匹配到规则



class MedicalRecordGenerator:
    """病历生成器 - 整合提取、校验、填充流程"""
    
    def __init__(self):
        self.extractor = InfoExtractor()
        self.validator = FactValidator()
        self.filler = TemplateFiller()
        self.classifier = InputTypeClassifier()
        self.language_detector = LanguageDetector()
    
    def generate(self, text: str, record_type: MedicalRecordType = MedicalRecordType.ADMISSION_NOTE) -> Tuple[MedicalRecord, Dict, str]:
        """生成病历主流程"""
        logger.info(f"开始生成{record_type.value}病历")
        
        # 第0a步：攻击意图检测
        is_suspicious, suspicious_msg = SuspiciousInputDetector.check(text)
        if is_suspicious:
            logger.warning(f"检测到疑似测试输入")
            record = MedicalRecord(record_type=record_type)
            record.chief_complaint = "请提供真实病情描述"
            record.present_illness = "待补充"
            record.past_history = "未提供"
            record.physical_exam = "待补充"
            record.auxiliary_exam = "本次无辅助检查"
            record.preliminary_diagnosis = ["请提供真实病情描述"]
            return record, {"warning": suspicious_msg, "input_type": "suspicious"}, "chinese"
        
        # 第0b步：语言检测（增强版）
        language, lang_error = self.language_detector.detect_mixed(text)
        if language != "chinese":
            logger.warning(f"语言检测拦截：{lang_error}")
            record = MedicalRecord(record_type=record_type)
            record.chief_complaint = lang_error
            record.present_illness = "待补充"
            record.past_history = "未提供"
            record.physical_exam = "待补充"
            record.auxiliary_exam = "本次无辅助检查"
            record.preliminary_diagnosis = ["待补充"]
            return record, {"error": "unsupported_language", "message": lang_error}, "english"
        
        # 第0c步：医学术语翻译（中英夹杂场景）
        translated_text = MedicalTermTranslator.translate(text)
        
        # 第1步：输入类型分类
        input_type = self.classifier.classify(translated_text)
        logger.info(f"输入类型：{input_type}")
        
        # 第2步：信息提取（数据模式跳过LLM避免幻觉）
        skip_llm = (input_type == "data")
        extracted_info = self.extractor.extract(translated_text, skip_llm=skip_llm)
        
        # 第2.5步：医学数值校验
        validated_info, value_warnings = MedicalValueValidator.validate_info(extracted_info)
        
        # 质控告警收集
        qc_warnings = []
        
        # 第3步：事实校验
        validated_info = self.validator.validate_info(validated_info, text)
        
        # 第3.5步：API 语义层（按需调用 DeepSeek）
        # 只有非 data 模式才调 API，节省 token
        api_cc_called = False
        api_diag_called = False
        if input_type != "data":
            api_client = APIClient()
            
            # 检测数据主导型混合输入：有实验室数据但无患者自述症状（仅有影像发现）
            _data_kw = ['AFP', 'CEA', 'CA199', 'CA153', '肿瘤标志物',
                        '凝血功能', '血气分析', '生化全套']
            _patient_symptom = re.search(r'(胸口痛|胸口疼|肚子疼|头痛|头晕|恶心|呕吐|'
                    r'腹泻|拉肚子|发烧|发热|咳嗽|咳痰|胸闷|气喘|心慌|'
                    r'心悸|乏力|没劲|水肿|皮疹|瘙痒|疼痛|感觉不舒服)', text)
            # 术后/复查场景不属于数据主导型混合输入
            _has_followup_context = re.search(r'(术后|复查|复诊|随访)\s*(\d|年|月|周)', text)
            _is_data_dominant = input_type == "mixed" and any(k in text for k in _data_kw) \
                and not _patient_symptom and not _has_followup_context
            api_cc = ""
            if _is_data_dominant:
                # 尝试生成更有意义的语义主诉，而非硬编码占位符
                scene_cc = _extract_data_cc(text)
                validated_info.chief_complaint = scene_cc if scene_cc else "患者提供检验数据"
            else:
                # 3.5a: 主诉生成 — 规则优先（零成本、零幻觉），LLM仅做增强
                rule_cc = build_cc_from_symptoms(
                    validated_info.positive_symptoms or [],
                    validated_info.negative_symptoms or [],
                    text
                )
                
                # 对话输入：规则CC不可靠（症状散布在问答中），优先LLM
                # 非对话输入：规则CC优先（零成本、零幻觉）
                if rule_cc and input_type != "dialogue":
                    validated_info.chief_complaint = rule_cc
                    api_cc_called = False
                    logger.info(f"规则CC: {rule_cc}")
                elif rule_cc and input_type == "dialogue":
                    # 对话输入：LLM优先，规则CC兜底
                    api_cc = api_client.generate_chief_complaint(validated_info, text)
                    if api_cc and not is_safe_phrase(api_cc) and len(api_cc) >= 4:
                        validated_info.chief_complaint = api_cc
                        api_cc_called = True
                        logger.info(f"对话LLM CC: {api_cc}")
                    else:
                        validated_info.chief_complaint = rule_cc
                        logger.info(f"对话LLM失败，回退规则CC: {rule_cc}")
                else:
                    # 无规则CC时调用API
                    api_cc = api_client.generate_chief_complaint(validated_info, text)
                    if api_cc and not is_safe_phrase(api_cc):
                        validated_info.chief_complaint = api_cc
                        api_cc_called = True
                        logger.info(f"API CC: {api_cc}")
                
                # 后续处理（安全套话检测等）— 仅当API返回了CC
                if api_cc_called and api_cc:
                    validated_info.chief_complaint = api_cc

                    # 规则CC兜底校验：API主诉若丢失关键症状，用规则版替代
                    _original_rule_cc = build_cc_from_symptoms(
                        validated_info.positive_symptoms or [],
                        validated_info.negative_symptoms or [],
                        text
                    )
                    rule_cc = _original_rule_cc
                    if rule_cc and rule_cc != api_cc:
                        # 检测API CC是否为症状描述、规则CC是否为纯病名
                        _disease_set = {"高血压","糖尿病","冠心病","慢阻肺","乙肝","甲亢","类风湿","肿瘤","癌症","肝炎","脂肪肝","肾病","胃炎","肺炎","支气管炎","贫血","阑尾炎","胰腺炎"}
                        api_is_symptom = not any(d in api_cc for d in _disease_set)
                        rule_is_disease = any(d in rule_cc for d in _disease_set) and not any(
                            s in rule_cc for s in ["痛","咳嗽","发热","烧","晕","肿","出血","外伤","吐","泻","喘"]
                        )
                        # 比较API主诉和规则主诉的症状覆盖度
                        api_symptom_count = sum(1 for s in (validated_info.positive_symptoms or []) if s in api_cc)
                        rule_symptom_count = sum(1 for s in (validated_info.positive_symptoms or []) if s in rule_cc)
                        # 规则版覆盖明显更多症状 → 替换（但API为症状描述时不替换）
                        if rule_symptom_count >= 2 and rule_symptom_count > api_symptom_count and not (api_is_symptom and rule_is_disease):
                            validated_info.chief_complaint = rule_cc
                            logger.info(f"规则CC覆盖更优: {api_cc!r} -> {rule_cc!r} (症状覆盖 {api_symptom_count} vs {rule_symptom_count})")
                    
                                                                                # 幻觉词检测：LLM CC中的词不在原文 → 编造，回退规则CC
                    _cc_hallucination_words = [chr(0x55dc)+chr(0x7761), chr(0x54b3)+chr(0x75f0)]  # 嗜睡, 咳痰
                    for _hw in _cc_hallucination_words:
                        if _hw in api_cc and _hw not in text and rule_cc:
                            validated_info.chief_complaint = rule_cc
                            logger.info(f"CC幻觉检测: 编造词'{_hw}'不在原文 -> 回退规则CC")
                            break
                    api_cc_called = True
                    logger.info(f"API 主诉生成: {api_cc}")
                    # API主诉安全套话检测 → 规则兜底
                    if is_safe_phrase(api_cc):
                        cc_fallback = generate_fallback_chief_complaint(text, validated_info)
                        if cc_fallback and cc_fallback != "主诉待补充":
                            validated_info.chief_complaint = cc_fallback
                            logger.info(f"API主诉安全套话被规则替代: {cc_fallback}")
                    # 复查/随访场景：规则主诉优先于API主诉
                    if not is_safe_phrase(api_cc):
                        rule_cc = generate_fallback_chief_complaint(text, validated_info)
                        if rule_cc and rule_cc not in ["主诉待补充", ""] and rule_cc != api_cc:
                            cc_keywords = ['复查', '复诊', '随访', '乳腺癌', '携带者']
                            if any(kw in text for kw in cc_keywords):
                                validated_info.chief_complaint = rule_cc
                                logger.info(f"复查场景规则主诉覆盖API: {api_cc} → {rule_cc}")
                    
                    # 慢性病复查场景覆盖：输入有"XX病X年"模式，但API生成冗长描述
                    if validated_info.chief_complaint and ('发现' in validated_info.chief_complaint or len(validated_info.chief_complaint) > 30):
                        chronic_diseases = re.findall(r'(高血压|糖尿病|冠心病|慢阻肺|乙肝|甲亢|类风湿|肾病|肿瘤)\s*(病)?\s*\d+', text)
                        if chronic_diseases and any(kw in text for kw in ['复查', '复诊', '开药', '拿药']):
                            disease_names = [d[0] for d in chronic_diseases]
                            simple_cc = "、".join(disease_names) + "复查"
                            validated_info.chief_complaint = simple_cc
                            logger.info(f"慢性病复查规则覆盖API: {api_cc} → {simple_cc}")
                        elif chronic_diseases and re.search(r'今日.*(?:血压|血糖|血脂|尿酸|糖化).*\d+', text):
                            disease_names = [d[0] for d in chronic_diseases]
                            simple_cc = "、".join(disease_names) + "复查"
                            validated_info.chief_complaint = simple_cc
                            logger.info(f"慢性病复查(今日检测)规则覆盖API: {api_cc} → {simple_cc}")
                    
                    # === LLM CC 后处理守卫 ===
                    logger.info(f"CC守卫进入: cc={validated_info.chief_complaint!r}, orig_rule={_original_rule_cc!r}")
                    
                    # 1. 慢性病名守卫：强制检测并替换纯慢性病名主诉
                    if validated_info.chief_complaint:
                        cc_stripped = re.sub(r'\s+', '', validated_info.chief_complaint)
                        disease_names = {'高血压', '糖尿病', '冠心病', '慢阻肺', '乙肝', '甲亢', '类风湿', '肿瘤', '癌症', '肝炎', '脂肪肝', '肾病'}
                        # 硬编码模式避免set顺序导致的regex构建问题
                        is_pure_disease = bool(cc_stripped and re.match(
                            r'^(?:高血压|糖尿病|冠心病|慢阻肺|乙肝|甲亢|类风湿|肿瘤|癌症|肝炎|脂肪肝|肾病)(?:[伴和、及](?:高血压|糖尿病|冠心病|慢阻肺|乙肝|甲亢|类风湿|肿瘤|癌症|肝炎|脂肪肝|肾病))*$',
                            cc_stripped))
                        
                        if is_pure_disease:
                            logger.warning(f'慢性病守卫触发: LLM CC={validated_info.chief_complaint!r}')
                            # 直接从原文提取症状
                            raw_pos, _ = InfoExtractor.extract_symptoms_safe(text)
                            real = [s for s in raw_pos if s not in disease_names]
                            if real:
                                tms = re.findall(r'(\d+\s*(?:天|周|个月|月|年|小时|分钟|日))', text)
                                ts = ''
                                for pf in ['天', '小时', '分钟', '周']:
                                    for t in tms:
                                        if pf in t:
                                            ts = t; break
                                    if ts: break
                                if not ts and tms: ts = tms[0]
                                new_cc = real[0] + ts
                                logger.warning(f'慢性病守卫: {validated_info.chief_complaint!r} -> {new_cc!r}')
                                validated_info.chief_complaint = new_cc
                            else:
                                logger.warning(f'慢性病守卫：无法从原文提取症状，保留原CC')
                        
                    # 2. "术后"守卫：文本含"术后"但LLM CC无"术后"，自动补入
                    if validated_info.chief_complaint:
                        has_postop_in_text = bool(re.search(r'术后', text))
                        has_postop_in_cc = '术后' in validated_info.chief_complaint
                        has_surgery_name = re.search(r'(乳腺癌|肝癌|胃癌|肺癌|结肠癌|直肠癌|甲状腺癌|胆囊|阑尾|剖宫产|子宫|卵巢|肾)', text)
                        if has_postop_in_text and not has_postop_in_cc and has_surgery_name:
                            postop_time = re.search(r'术后\s*(\d+\s*(?:年|个月|月|周|天))', text)
                            time_suffix = postop_time.group(1) if postop_time else ""
                            if time_suffix:
                                validated_info.chief_complaint = validated_info.chief_complaint.replace('复查', f'术后{time_suffix}复查')
                                if '术后' not in validated_info.chief_complaint:
                                    validated_info.chief_complaint = validated_info.chief_complaint.replace('复查', f'术后{time_suffix}复查')
                            else:
                                validated_info.chief_complaint = validated_info.chief_complaint.replace('复查', '术后复查')
                            logger.info(f"术后守卫: 补入术后 -> {validated_info.chief_complaint!r}")
                    
                    # 3. 照抄原文守卫：LLM CC长度>25字且含大量口语词 → 回退
                    if validated_info.chief_complaint:
                        colloquial_in_cc = ['不舒服', '不得劲', '吃饭不香', '睡觉不好', '我觉得', '感觉']
                        cc_len = len(validated_info.chief_complaint)
                        has_colloquial = any(w in validated_info.chief_complaint for w in colloquial_in_cc)
                        if cc_len > 25 and has_colloquial:
                            guard_cc = _original_rule_cc if _original_rule_cc not in ["主诉待补充", ""] else ""
                            if guard_cc:
                                validated_info.chief_complaint = guard_cc
                                logger.info(f"照抄守卫: LLM CC过长且口语 -> 规则CC {rule_cc!r}")

                    # API主诉后处理：过滤被否定症状（NEG系列保护）
                    if validated_info.chief_complaint and validated_info.negative_symptoms:
                        for neg_sym in validated_info.negative_symptoms:
                            if neg_sym in validated_info.chief_complaint:
                                validated_info.chief_complaint = validated_info.chief_complaint.replace(neg_sym, "")
                        validated_info.chief_complaint = re.sub(r'[、，,]+', '、', validated_info.chief_complaint).strip('、，, ')
                        if not validated_info.chief_complaint:
                            if validated_info.positive_symptoms:
                                validated_info.chief_complaint = "、".join(validated_info.positive_symptoms)[:30]
                            else:
                                validated_info.chief_complaint = "待补充"
            
# 3.5b: 诊断推断 - 急症快检 + 结构化LLM（重构版）
            # 3.5b: 诊断推断 - 急症快检 + 结构化LLM
            # 急症快速检查（10个高危急症，基于明确体征/检验组合）
            emergency_diag = _check_emergency_diagnosis(text, validated_info)
            if emergency_diag:
                validated_info.disease_type = emergency_diag
                logger.info(f"急症诊断直接采用: {emergency_diag}")
            else:
                # 非急症：结构化LLM一步诊断
                api_diag = api_client.infer_diagnosis(validated_info, text)
                # 后处理：翻译残留英文术语（如STEMI→急性ST段抬高型心肌梗死）
                if api_diag:
                    api_diag = MedicalTermTranslator.translate(api_diag)
                    

                if api_diag and not is_safe_phrase(api_diag) and len(api_diag) >= 4:
                    validated_info.disease_type = api_diag
                    api_diag_called = True
                    logger.info(f"LLM诊断: {api_diag}")
                elif not validated_info.disease_type:
                    # LLM失败，尝试纯规则常见病诊断
                    common_diag = _check_common_diagnosis(text, validated_info)
                    if common_diag:
                        validated_info.disease_type = common_diag
                        logger.info(f"规则常见病诊断: {common_diag}")
                    else:
                        validated_info.disease_type = "待补充"
        
        # 数据模式：独立运行急诊快检+常见病规则（避免data类型跳过诊断）
        if input_type == "data":
            emergency_diag = _check_emergency_diagnosis(text, validated_info)
            if emergency_diag:
                validated_info.disease_type = emergency_diag
                logger.info(f"数据模式急症诊断: {emergency_diag}")
            elif not validated_info.disease_type or validated_info.disease_type == "待补充":
                common_diag = _check_common_diagnosis(text, validated_info)
                if common_diag:
                    validated_info.disease_type = common_diag
                    logger.info(f"数据模式规则诊断: {common_diag}")
        
        # 主诉后处理：规范化+去污染
        if validated_info.chief_complaint:
            cc = validated_info.chief_complaint
            # 1. 规范化"X月"→"X个月"
            cc = re.sub(r'(\d+)月(?!个)', r'\1个月', cc)
                        # 2. 去重：清理LLM输出的连续重复字符
            # "咽疼痛痛"→"咽疼痛"→"咽痛"、"咳嗽嗽"→"咳嗽"
            cc = re.sub(r'([一-鿿])\1+', r'\1', cc)
            # 3. 去除LLM输出的古怪字符
            cc = cc.replace('\u200b', '').replace('\u200d', '')  # 零宽字符
            # 4. 截断至15字
            validated_info.chief_complaint = cc
            # 截断至15字
            raw_cc = validated_info.chief_complaint
            if len(raw_cc) > 15:
                validated_info.chief_complaint = raw_cc[:15].rstrip('，、。，；,')
                # 诊断方向合理性检查（文书质量控制）
        if validated_info.disease_type and validated_info.disease_type != "待补充":
            dir_warning = DirectionGuard.check(
                validated_info.chief_complaint or "",
                validated_info.disease_type,
                text
            )
            if dir_warning:
                qc_warnings.append(dir_warning)
        
# 第4步：模板填充（传入输入类型和原始文本）
        if record_type == MedicalRecordType.FOLLOW_UP_RECORD:
            record = self.filler.fill_follow_up_record(validated_info, input_type, text)
        elif record_type == MedicalRecordType.DISCHARGE_SUMMARY:
            record = self.filler.fill_discharge_summary(validated_info, input_type, text)
        else:
            record = self.filler.fill_admission_note(validated_info, input_type, text)
        
        # 生成质控信息
        qc_info = self._generate_qc_info(validated_info, text)
        qc_info["input_type"] = input_type
        qc_info["api_called"] = {"chief_complaint": api_cc_called, "diagnosis": api_diag_called}
        if value_warnings:
            qc_info["value_warnings"] = value_warnings
        if qc_warnings:
            qc_info["direction_warnings"] = qc_warnings
        
        return record, qc_info, MedicalRecordAgent.format_record(record, text)
    
    def _generate_qc_info(self, info: ExtractedInfo, text: str) -> Dict:
        """生成质控信息"""
        issues = []
        
        if not info.chief_complaint:
            issues.append({
                "severity": "high",
                "category": "missing",
                "message": "主诉缺失",
                "suggestion": "请补充主诉（症状+时长）"
            })
        
        if not info.gender:
            issues.append({
                "severity": "medium",
                "category": "missing",
                "message": "性别信息缺失",
                "suggestion": "请补充患者性别"
            })
        
        if not info.age:
            issues.append({
                "severity": "medium",
                "category": "missing",
                "message": "年龄信息缺失",
                "suggestion": "请补充患者年龄"
            })
        
        return {
            "extracted_fields": {
                "gender": bool(info.gender),
                "age": bool(info.age),
                "chief_complaint": bool(info.chief_complaint),
                "blood_sugar": bool(info.blood_sugar),
                "blood_sugar_2h": bool(info.blood_sugar_2h),
                "hba1c": bool(info.hba1c),
                "blood_pressure": bool(info.blood_pressure),
                "symptoms": len(info.symptoms) > 0
            },
            "issues": issues,
            "extracted_info": asdict(info)
        }


class MedicalRecordAgent:
    """智能体主类"""
    
    def __init__(self):
        self.generator = MedicalRecordGenerator()
    
    # ── 输入守卫：拦截恶意/异常/无意义输入 ──────────────────────────────
    # 返回 (is_valid, block_reason, block_message)
    _GUARD_MESSAGES = {
        "empty":       "请输入医患对话或检验数据",
        "too_long":    "输入过长（超过10000字符），请精简后重试",
        "only_special":"未能识别有效医学信息，请输入中文描述患者病情",
        "only_digits": "未能识别有效医学信息，请输入中文描述患者病情",
        "repeated":    "输入内容无意义，请输入有效的患者病情描述",
        "keyboard_smash":"输入内容无意义，请输入有效的患者病情描述",
        "script_injection":"输入包含非法内容，请输入有效的患者病情描述",
        "sql_injection":"输入包含非法内容，请输入有效的患者病情描述",
        "control_chars":"输入包含非法字符，请输入有效的患者病情描述",
        "non_medical_en":"暂不支持非医疗英文输入，请使用中文描述患者情况",
        "random_garbage":"未能识别有效医学信息，请输入中文描述患者病情",
        "emoji_only":  "未能识别有效医学信息，请输入中文描述患者病情",
        "too_few_chinese":"输入的中文内容过少，请输入有效的患者病情描述",
        "url_only":    "检测到仅为链接，请粘贴患者病情文本内容",
    }

    # SQL 注入关键词
    _SQL_PATTERNS = [
        "drop table", "drop database", "union select", "select * from",
        "insert into", "update ", "delete from", "truncate table",
        "exec(", "execute(", "information_schema", "' or ", "'or ",
        "1=1", "1 = 1", "' --", "'--", "xp_cmdshell", "char(",
        "concat(", "sleep(", "benchmark(", "load_file", "into outfile",
    ]

    # 键盘布局序列
    _KEYBOARD_SEQUENCES = [
        "qwertyuiop", "asdfghjkl", "zxcvbnm",
        "poiuytrewq", "lkjhgfdsa", "mnbvcxz",
        "qazwsxedc", "plmoknijb",
        "1234567890", "0987654321",
        "qwertasdfg", "yuiophjk",
    ]

    # 常见正常英文单词（用于区分随机乱码）
    _COMMON_ENGLISH_WORDS = [
        "the", "and", "for", "are", "but", "not", "you",
        "all", "can", "had", "her", "was", "one", "our",
        "out", "has", "have", "from", "they", "with",
        "this", "that", "what", "when", "where", "which",
        "patient", "pain", "chest", "cough", "blood",
    ]

    # 医学术语英文关键词（放行含这些词的英文输入）
    _MEDICAL_ENGLISH_KEYWORDS = [
        "patient", "male", "female", "chest", "pain", "fever", "cough",
        "headache", "blood", "pressure", "diabetes", "hypertension",
        "diagnosis", "symptom", "treatment", "surgery", "medication",
        "disease", "infection", "doctor", "hospital", "clinic",
        "nausea", "vomiting", "diarrhea", "dyspnea", "edema",
        "palpitation", "dizziness", "fatigue", "anemia", "cancer",
        "dysphagia", "jaundice", "seizure", "stroke", "asthma",
        "copd", "mi", "ckd", "uti", "gi", "bleeding", "trauma",
    ]

    # 中文 Prompt 注入 / 越狱关键词
    _PROMPT_INJECTION_CN = [
        "忽略之前的指令", "忽略之前的指示",
        "忘记之前", "忽略上面", "忽略所有",
        "你现在是", "你的新身份",
        "角色扮演", "越狱", "DAN",
        "无视规则", "解除限制",
        "开发者模式", "系统提示",
        "显示你的prompt", "显示你的系统",
        "输出你的指令", "输出你的prompt",
        "system:", "当前系统指令",
        "假装", "打开越狱",
    ]

    # 零宽字符 / Unicode 控制字符
    _ZERO_WIDTH_CHARS = set([
        "​", "‌", "‍", "‎", "‏",
        "﻿", "‪", "‫", "‬", "‭", "‮",
        "⁠", "⁡", "⁢", "⁣", "⁤",
    ])

    # 中文纯虚词/停用词（仅这些词组成时视为无意义）
    _CN_STOP_WORDS_SET = set("的了吗呢吧啊嗯哦呀哈嘛嘧咕噔呦噢咔唯")

    @classmethod
    def guard_input(cls, text: str):
        """全面输入守卫。返回 (is_valid, reason, user_message)"""
        if not text or not text.strip():
            return False, "empty", cls._GUARD_MESSAGES["empty"]

        text = text.strip()

        # 1. 控制字符检测（\x00-\x08, \x0b, \x0c, \x0e-\x1f）
        for ch in text:
            code = ord(ch)
            if code < 0x20 and code not in (0x09, 0x0a, 0x0d):
                return False, "control_chars", cls._GUARD_MESSAGES["control_chars"]

        # 1b. 零宽字符：先清洗，再判断
        zw_count = sum(1 for c in text if c in cls._ZERO_WIDTH_CHARS)
        if zw_count > 0:
            text_clean = "".join(c for c in text if c not in cls._ZERO_WIDTH_CHARS)
            # 纯零宽字符 → 拦截
            if len(text_clean.strip()) < 3:
                return False, "control_chars", cls._GUARD_MESSAGES["control_chars"]
            # 零宽字符占比超过 80% → 拦截
            if zw_count >= max(len(text) * 0.8, 20):
                return False, "control_chars", cls._GUARD_MESSAGES["control_chars"]
            # 少量零宽字符夹带 → 清洗后继续
            text = text_clean

        # 2. HTML/script 注入检测
        text_lower = text.lower()
        script_pats = [
            "<script", "</script", "javascript:", "onerror=", "onclick=",
            "onload=", "onmouseover=", "<iframe", "<img ", "<svg ",
            "<object", "<embed", "<link ", "data:text/html",
            "vbscript:", "eval(", "document.cookie", "document.write",
            "window.location", "settimeout(", "setinterval(",
        ]
        for pat in script_pats:
            if pat in text_lower:
                return False, "script_injection", cls._GUARD_MESSAGES["script_injection"]

        # 2b. 中文 Prompt 注入 / 越狱检测
        for pat in cls._PROMPT_INJECTION_CN:
            if pat in text_lower:
                return False, "script_injection", cls._GUARD_MESSAGES["script_injection"]

        # 3. SQL 注入检测
        for pat in cls._SQL_PATTERNS:
            if pat in text_lower:
                return False, "sql_injection", cls._GUARD_MESSAGES["sql_injection"]

        # 4. 超长输入
        if len(text) > 10000:
            return False, "too_long", cls._GUARD_MESSAGES["too_long"]

        # 5. 过短（<3个非空白字符）
        no_space = text.replace(" ", "").replace("\n", "").replace("\t", "")
        if len(no_space) < 3:
            return False, "empty", cls._GUARD_MESSAGES["empty"]

        # 6. 字符分类统计
        chinese_chars = [c for c in text if "\u4e00" <= c <= "\u9fff"]
        digit_chars = [c for c in text if c.isdigit()]
        alpha_chars = [c for c in text if c.isalpha()]
        n_cn = len(chinese_chars)
        n_digit = len(digit_chars)
        n_alpha = len(alpha_chars)
        total_meaningful = n_cn + n_digit + n_alpha

        # 7. 纯特殊字符（无字母、数字、中文）
        if total_meaningful == 0:
            return False, "only_special", cls._GUARD_MESSAGES["only_special"]

        # 8. Emoji 轰炸
        emoji_cnt = sum(1 for c in text if ord(c) > 0x1F000 or (0x2600 <= ord(c) <= 0x27BF))
        clean_len = len(text.replace(" ", ""))
        if clean_len > 0 and emoji_cnt >= clean_len * 0.7 and total_meaningful <= 2:
            return False, "emoji_only", cls._GUARD_MESSAGES["emoji_only"]

        # 9. 纯 URL
        if text.startswith(("http://", "https://", "ftp://", "file://")) and n_cn == 0:
            return False, "url_only", cls._GUARD_MESSAGES["url_only"]

        # 10. 无中文分支
        if n_cn == 0:
            # 纯数字
            if n_digit > 0 and n_alpha == 0:
                return False, "only_digits", cls._GUARD_MESSAGES["only_digits"]

            clean_alpha = "".join(c for c in text_lower if c.isalpha())
            if len(clean_alpha) >= 6:
                # 键盘序列检测
                for seq in cls._KEYBOARD_SEQUENCES:
                    for w in range(4, min(len(seq) + 1, len(clean_alpha) + 1)):
                        for i in range(len(seq) - w + 1):
                            if seq[i:i + w] in clean_alpha:
                                return False, "keyboard_smash", cls._GUARD_MESSAGES["keyboard_smash"]
                # 重复字符
                if len(clean_alpha) >= 10 and len(set(clean_alpha)) <= 3:
                    return False, "keyboard_smash", cls._GUARD_MESSAGES["keyboard_smash"]

            # 随机乱码检测（高唯一字符比 + 无真实单词）
            if len(clean_alpha) >= 8:
                unique_ratio = len(set(clean_alpha)) / len(clean_alpha)
                if unique_ratio > 0.85:
                    has_real = any(w in text_lower.split() for w in cls._COMMON_ENGLISH_WORDS)
                    if not has_real:
                        return False, "random_garbage", cls._GUARD_MESSAGES["random_garbage"]

            # 非医疗英文
            if n_alpha >= 5:
                if not any(kw in text_lower for kw in cls._MEDICAL_ENGLISH_KEYWORDS):
                    return False, "non_medical_en", cls._GUARD_MESSAGES["non_medical_en"]

        # 11. 中文内容质量检测
        if n_cn > 0:
            unique_cn = len(set(chinese_chars))
            # 纯虚词/停用词检测
            if n_cn >= 3:
                stop_ratio = sum(1 for c in chinese_chars if c in cls._CN_STOP_WORDS_SET) / n_cn
                if stop_ratio > 0.8:
                    return False, "repeated", cls._GUARD_MESSAGES["repeated"]
            is_measurement = (n_cn > 0 and n_digit > 0)
            if unique_cn < 3:
                if not is_measurement and n_alpha + n_digit > n_cn * 3:
                    return False, "too_few_chinese", cls._GUARD_MESSAGES["too_few_chinese"]
                if n_cn >= 8 and not is_measurement:
                    return False, "repeated", cls._GUARD_MESSAGES["repeated"]
            if n_cn >= 10 and unique_cn <= 2 and not is_measurement:
                return False, "repeated", cls._GUARD_MESSAGES["repeated"]

        return True, "", ""


    def process_input(self, input_text: str, record_type: str = "admission_note") -> Dict:
        # Step 1: full input guard
        is_valid, reason, block_msg = MedicalRecordAgent.guard_input(input_text)
        if not is_valid:
            rec = MedicalRecord(record_type=MedicalRecordType.ADMISSION_NOTE)
            rec.chief_complaint = block_msg
            rec.present_illness = "待补充"
            rec.past_history = "未提供"
            rec.personal_history = "待补充"
            rec.family_history = "待补充"
            rec.physical_exam = "- 一般情况：待补充，待补充岁\n- 生命体征：T待补充℃，P待补充次/分，R待补充次/分，BP待补充\n- 其他：待补充"
            rec.auxiliary_exam = "本次无辅助检查"
            rec.preliminary_diagnosis = ["待补充"]
            rec.treatment_plan = "待补充"
            return {
                "status": "success",
                "record": rec,
                "qc_info": {"input_guarded": True, "guard_reason": reason},
                "formatted_output": MedicalRecordAgent.format_record(rec),
                "disclaimer": "本智能体是医生的辅助工具，不具备独立诊断权，所有输出仅供参考。"
            }

        text = input_text.strip()

        # 超长输入截断（守卫已拦截>10000，此处控制令牌消耗）
        if len(text) > 2000:
            text = text[:2000]
            logger.info(f"输入超2000字，已截取前2000字处理")

        # 自动识别病历类型
        type_map = {
            "admission_note": MedicalRecordType.ADMISSION_NOTE,
            "outpatient_note": MedicalRecordType.OUTPATIENT_NOTE,
            "discharge_summary": MedicalRecordType.DISCHARGE_SUMMARY,
            "follow_up_record": MedicalRecordType.FOLLOW_UP_RECORD
        }
        
        # 如果显式指定了非默认类型，直接使用（避免关键词覆盖）
        if record_type != "admission_note":
            record_type_enum = type_map.get(record_type, MedicalRecordType.ADMISSION_NOTE)
        else:
            text_lower = input_text.lower()
            # 根据关键词自动选择模板（仅在默认admission_note时）
            if "外院诊断" in text_lower:
                record_type_enum = MedicalRecordType.ADMISSION_NOTE
            elif "随访" in text_lower:
                record_type_enum = MedicalRecordType.FOLLOW_UP_RECORD
            elif "复查" in text_lower and "外院诊断" not in text_lower:
                record_type_enum = MedicalRecordType.FOLLOW_UP_RECORD
            elif "出院" in text_lower:
                record_type_enum = MedicalRecordType.DISCHARGE_SUMMARY
            else:
                record_type_enum = MedicalRecordType.ADMISSION_NOTE
        

        record, qc_info, language = self.generator.generate(input_text, record_type_enum)

        if language != "chinese":
            return {
                "status": "success",
                "record": record,
                "qc_info": qc_info,
                "formatted_output": MedicalRecordAgent.format_record(record),
                "disclaimer": "⚠️ 本智能体是医生的辅助工具，不具备独立诊断权，所有输出仅供参考。"
            }
        return {
            "status": "success",
            "record": record,
            "qc_info": qc_info,
            "record_type": record_type_enum.value,
            "formatted_output": MedicalRecordAgent.format_record(record),
            "disclaimer": "⚠️ 本智能体是医生的辅助工具，不具备独立诊断权，所有输出仅供参考。"
        }
    
    @staticmethod
    def format_record(record: MedicalRecord, raw_input: str = "") -> str:
        """格式化病历输出（标准格式），包含隐私脱敏"""
        output = []
        
        # 对输入文本进行脱敏
        desensitized_input = PrivacyDesensitizer.desensitize_all(raw_input)
        
        if record.record_type == MedicalRecordType.FOLLOW_UP_RECORD:
            output.append(f"# 慢病随访记录 [{record.record_id}]")
            output.append("")
            output.append(f"**主诉：** {record.chief_complaint}")
            output.append("")
            output.append(f"**现病史：** {record.present_illness}")
            output.append("")
            output.append("**体格检查：**")
            output.append(record.physical_exam or "待补充")
            output.append("")
            output.append(f"**辅助检查：** {record.auxiliary_exam}")
            output.append("")
            if record.personal_history and record.personal_history != "待补充":
                output.append(f"**随访评估：** {record.personal_history}")
                output.append("")
            output.append(f"**初步诊断：** {''.join(record.preliminary_diagnosis)}")
            output.append("")
            output.append(f"**治疗计划：** {record.treatment_plan or '待补充'}")
            output.append("")
            if record.family_history and record.family_history != "待补充":
                output.append(f"**下次随访：** {record.family_history}")
                output.append("")
        elif record.record_type == MedicalRecordType.DISCHARGE_SUMMARY:
            output.append(f"# 出院小结 [{record.record_id}]")
            output.append("")
            output.append(f"**主诉：** {record.chief_complaint}")
            output.append("")
            output.append(f"**入院情况：** {record.present_illness}")
            output.append("")
            output.append(f"**诊疗经过：** {record.past_history or '详见入院记录'}")
            output.append("")
            output.append("**体格检查：**")
            output.append(record.physical_exam or "待补充")
            output.append("")
            output.append(f"**辅助检查：** {record.auxiliary_exam}")
            output.append("")
            output.append("**出院诊断：**")
            for i, diagnosis in enumerate(record.preliminary_diagnosis, 1):
                output.append(f"{i}. {diagnosis}")
            output.append("")
            output.append(f"**出院医嘱：** {record.treatment_plan or '待补充'}")
            output.append("")
            if record.family_history and record.family_history != "待补充":
                output.append(f"**{record.family_history}**")
                output.append("")
            if record.personal_history and record.personal_history != "待补充":
                output.append(f"**随访建议：** {record.personal_history}")
                output.append("")
        else:
            # 标准入院记录格式
            output.append(f"# 入院记录 [{record.record_id}]")
            output.append("")
            # 患者基本信息（放在最前面）
            if record.patient_info:
                output.append(record.patient_info)
                output.append("")
            output.append(f"**主诉：** {record.chief_complaint}")
            output.append("")
            output.append(f"**现病史：** {record.present_illness}")
            output.append("")
            output.append(f"**既往史：** {record.past_history or '待补充'}")
            output.append("")
            output.append(f"**个人史、婚育史、家族史：** {record.personal_history or '待补充'}")
            output.append("")
            output.append("**体格检查：**")
            output.append(record.physical_exam)
            output.append("")
            output.append(f"**辅助检查：** {record.auxiliary_exam}")
            output.append("")
            output.append("**初步诊断：**")
            for i, diagnosis in enumerate(record.preliminary_diagnosis, 1):
                output.append(f"{i}. {diagnosis}")
            output.append("")
            output.append(f"**治疗计划：** {record.treatment_plan or '待补充'}")
        
        output_text = "\n".join(output)
        # 对整个输出应用脱敏处理（双重保险）
        output_text = PrivacyDesensitizer.desensitize(output_text)
        
        return output_text






