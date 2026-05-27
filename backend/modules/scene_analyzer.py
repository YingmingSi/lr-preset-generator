"""
场景识别与AI校正模块
支持用户手动指定参考图场景和原图场景
检测场景不匹配冲突并给出针对性提示
"""

import anthropic
import base64
import json
from PIL import Image
import io
from typing import Optional


SCENE_RULES = {
    'portrait': {
        'HueAdjustmentOrange':        lambda v: clamp(v, -5,  5),
        'HueAdjustmentRed':           lambda v: clamp(v, -5,  5),
        'SaturationAdjustmentOrange': lambda v: clamp(v, -20, 15),
        'SaturationAdjustmentRed':    lambda v: clamp(v, -20, 10),
        'LuminanceAdjustmentOrange':  lambda v: clamp(v, -10, 40),
        'ColorGradeShadowLum':        lambda v: clamp(v, -10, 20),
        'Clarity':                    lambda v: clamp(v, -30, 15),
        'Dehaze':                     lambda v: 0,
        'max_sat_global':             45,
        'needs_portrait_mask':        True,
    },
    'landscape': {
        'SaturationAdjustmentBlue':   lambda v: clamp(v, -40, 40),
        'SaturationAdjustmentGreen':  lambda v: clamp(v, -40, 20),
        'HueAdjustmentGreen':         lambda v: clamp(v, -20, 20),
        'Dehaze':                     lambda v: clamp(v,   0, 20),
        'max_sat_global':             70,
    },
    'architecture': {
        'Clarity':                    lambda v: clamp(v, -10, 40),
        'SaturationAdjustmentBlue':   lambda v: clamp(v, -30, 30),
        'Dehaze':                     lambda v: clamp(v,   0, 15),
        'max_sat_global':             65,
    },
    'still_life': {
        'SaturationAdjustmentOrange': lambda v: clamp(v, -10, 35),
        'SaturationAdjustmentYellow': lambda v: clamp(v, -10, 30),
        'LuminanceAdjustmentOrange':  lambda v: clamp(v,  -5, 30),
        'Dehaze':                     lambda v: 0,
        'max_sat_global':             70,
    },
    'night': {
        'Shadows':                    lambda v: clamp(v, -20, 40),
        'Blacks':                     lambda v: clamp(v, -40, 10),
        'ColorNoiseReduction':        lambda v: max(v, 30),
        'Dehaze':                     lambda v: clamp(v,   0, 15),
        'max_sat_global':             60,
    },
    'other': {
        'max_sat_global': 70,
    },
}

# 场景不匹配时需要重点检查的参数
SCENE_CONFLICT_WARNINGS = {
    ('landscape', 'portrait'): {
        'message': '⚠️ 参考图为风光风格，原图为人像。橙色/红色参数已自动收窄，建议重点检查肤色的色相、饱和度和明度。',
        'extra_rules': {
            'HueAdjustmentOrange':        lambda v: clamp(v, -5,  5),
            'SaturationAdjustmentOrange': lambda v: clamp(v, -15, 10),
            'LuminanceAdjustmentOrange':  lambda v: clamp(v, -5,  35),
        },
    },
    ('architecture', 'portrait'): {
        'message': '⚠️ 参考图为建筑/城市风格，原图为人像。肤色参数已自动保护，建议检查橙色色相还原。',
        'extra_rules': {
            'HueAdjustmentOrange':        lambda v: clamp(v, -5,  5),
            'SaturationAdjustmentOrange': lambda v: clamp(v, -20, 12),
        },
    },
    ('night', 'portrait'): {
        'message': '⚠️ 参考图为夜景风格，原图为人像。整体饱和度已收窄，建议检查肤色自然度。',
        'extra_rules': {
            'SaturationAdjustmentOrange': lambda v: clamp(v, -20, 12),
        },
    },
    ('portrait', 'landscape'): {
        'message': 'ℹ️ 参考图为人像风格，原图为风光。色彩调整幅度偏保守，绿色/蓝色可适当放开。',
        'extra_rules': {},
    },
}

STYLE_COLOR_GRADING = {
    '青橙': {
        'SplitToningShadowHue':           195,
        'SplitToningShadowSaturation':    20,
        'SplitToningHighlightHue':        35,
        'SplitToningHighlightSaturation': 15,
    },
    '日系': {
        'SplitToningShadowHue':           170,
        'SplitToningShadowSaturation':    15,
        'SplitToningHighlightHue':        55,
        'SplitToningHighlightSaturation': 10,
    },
    '港风': {
        'SplitToningShadowHue':           190,
        'SplitToningShadowSaturation':    25,
        'SplitToningHighlightHue':        75,
        'SplitToningHighlightSaturation': 15,
    },
    '黑金': {
        'SplitToningShadowHue':           0,
        'SplitToningShadowSaturation':    5,
        'SplitToningHighlightHue':        40,
        'SplitToningHighlightSaturation': 25,
    },
    '莫兰迪': {
        'SplitToningShadowHue':           200,
        'SplitToningShadowSaturation':    8,
        'SplitToningHighlightHue':        50,
        'SplitToningHighlightSaturation': 8,
    },
    '赛博朋克': {
        'SplitToningShadowHue':           240,
        'SplitToningShadowSaturation':    30,
        'SplitToningHighlightHue':        300,
        'SplitToningHighlightSaturation': 20,
    },
    '胶片': {
        'SplitToningShadowHue':           210,
        'SplitToningShadowSaturation':    12,
        'SplitToningHighlightHue':        45,
        'SplitToningHighlightSaturation': 10,
    },
}

SCENE_LABELS = {
    'portrait':     '人像',
    'landscape':    '风光',
    'architecture': '建筑 / 城市',
    'still_life':   '静物',
    'night':        '夜景',
    'other':        '其他',
}


def analyze_scene_and_correct(
    ref_image_bytes: bytes,
    raw_params: dict,
    src_image_bytes: Optional[bytes] = None,
    ref_scene_type: str = 'auto',
    src_scene_type: str = 'auto',
) -> dict:
    """
    场景识别 + 参数校正

    Args:
        ref_image_bytes: 参考图字节
        raw_params:      初步分析参数
        src_image_bytes: 原图字节（可选）
        ref_scene_type:  参考图场景（用户指定或auto）
        src_scene_type:  原图场景（用户指定或auto）
    """
    # 1. 识别参考图场景
    if ref_scene_type != 'auto':
        scene_info = _build_manual_scene(ref_scene_type)
        style_info = _recognize_style_only(ref_image_bytes)
        scene_info.update(style_info)
    else:
        scene_info = _recognize_scene(ref_image_bytes)

    # 2. 确定原图场景（用于冲突检测）
    src_scene = src_scene_type if src_scene_type != 'auto' else None

    # 3. 按参考图场景应用规则
    corrected = _apply_scene_rules(raw_params, scene_info['scene_type'])

    # 4. 中间调饱和度全局限制
    corrected['ColorGradeMidtoneSat'] = clamp(corrected.get('ColorGradeMidtoneSat', 0), -10, 10)
    corrected['ColorGradeMidtoneHue'] = 0

    # 5. 风格先验：颜色分级
    corrected = _apply_style_prior(corrected, scene_info.get('style_keywords', []))

    # 6. 场景冲突检测
    conflict_info = _detect_conflict(scene_info['scene_type'], src_scene, corrected)

    # 7. Claude参数校正
    corrected = _claude_param_correction(ref_image_bytes, corrected, scene_info)

    # 8. 置信度和报告
    confidence = _compute_confidence(corrected, scene_info, ref_scene_type)
    report     = _generate_report(scene_info, corrected, confidence, conflict_info, src_scene)

    return {
        'params':     corrected,
        'scene_info': scene_info,
        'confidence': confidence,
        'report':     report,
    }


def _build_manual_scene(scene_type: str) -> dict:
    return {
        'scene_type':       scene_type,
        'has_face':         scene_type == 'portrait',
        'dominant_subject': SCENE_LABELS.get(scene_type, scene_type),
        'style_keywords':   [],
        'estimated_style':  '',
        'lighting':         '',
        'grain_visible':    False,
        'special_notes':    '',
        'api_success':      True,
        'user_specified':   True,
    }


def _recognize_scene(image_bytes: bytes) -> dict:
    """完整AI场景识别"""
    client  = anthropic.Anthropic()
    img_b64 = _compress_for_api(image_bytes)

    prompt = """分析这张图片，返回JSON格式的场景信息。

场景类型判断规则：
- "portrait"：能清楚看到人脸，人物占画面主体
- "landscape"：自然风光为主，人物占比很小或无人
- "architecture"：建筑/城市/街头为主，无明显人脸
- "still_life"：静物类，包括食物/产品/花卉/桌面摆拍等
- "night"：夜景或明显低光环境
- "other"：以上都不符合

请识别：
1. scene_type: 从上述六类中选一个
2. has_face: 是否能清楚看到人脸（true/false）
3. dominant_subject: 主要拍摄对象（一句话）
4. style_keywords: 风格关键词列表（如["冷调","低饱和","电影感"]）
5. estimated_style: 估计的修图风格（如"青橙电影感"）
6. lighting: 光线描述
7. grain_visible: 是否有明显胶片颗粒感（true/false）
8. special_notes: 需要注意的特殊情况

只返回JSON，不要其他文字。"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text",  "text": prompt}
                ]
            }]
        )
        text = response.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        result = json.loads(text)
        result['api_success']   = True
        result['user_specified'] = False
        return result
    except Exception as e:
        return {
            'scene_type': 'other', 'has_face': False,
            'dominant_subject': '未知', 'style_keywords': [],
            'estimated_style': '未知', 'lighting': '未知',
            'grain_visible': False, 'special_notes': '',
            'api_success': False, 'user_specified': False, 'error': str(e),
        }


def _recognize_style_only(image_bytes: bytes) -> dict:
    """用户已指定场景时，只识别风格关键词"""
    client  = anthropic.Anthropic()
    img_b64 = _compress_for_api(image_bytes)

    prompt = """分析这张图片的修图风格，返回JSON：
{
  "style_keywords": ["风格关键词列表"],
  "estimated_style": "整体风格描述",
  "lighting": "光线描述",
  "grain_visible": false
}
只返回JSON。"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text",  "text": prompt}
                ]
            }]
        )
        text = response.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception:
        return {'style_keywords': [], 'estimated_style': '', 'lighting': '', 'grain_visible': False}


def _detect_conflict(ref_scene: str, src_scene: Optional[str], params: dict) -> dict:
    """检测参考图和原图场景不匹配，应用额外约束"""
    if not src_scene or src_scene == ref_scene:
        return {'has_conflict': False, 'message': None}

    conflict_key = (ref_scene, src_scene)
    conflict     = SCENE_CONFLICT_WARNINGS.get(conflict_key)

    if not conflict:
        return {'has_conflict': False, 'message': None}

    # 应用额外的参数约束
    for key, rule in conflict.get('extra_rules', {}).items():
        if key in params and callable(rule):
            params[key] = rule(params[key])

    return {
        'has_conflict': True,
        'message':      conflict['message'],
        'ref_scene':    ref_scene,
        'src_scene':    src_scene,
    }


def _apply_style_prior(params: dict, style_keywords: list) -> dict:
    """根据风格关键词用先验库校正颜色分级（各50%加权）"""
    for keyword in style_keywords:
        for style_key, grading in STYLE_COLOR_GRADING.items():
            if style_key in keyword:
                for param, prior_value in grading.items():
                    current      = params.get(param, 0)
                    params[param] = int(current * 0.5 + prior_value * 0.5)
                break
    return params


def _claude_param_correction(image_bytes: bytes, params: dict, scene_info: dict) -> dict:
    """Claude语义层面参数校正"""
    client  = anthropic.Anthropic()
    img_b64 = _compress_for_api(image_bytes)

    key_params = {
        'HSL色相':  {k: v for k, v in params.items() if k.startswith('HueAdjustment')},
        'HSL饱和度': {k: v for k, v in params.items() if k.startswith('SaturationAdjustment')},
        'HSL明度':  {k: v for k, v in params.items() if k.startswith('LuminanceAdjustment')},
        '亮度':     {k: params.get(k) for k in ['Exposure','Contrast','Highlights','Shadows','Whites','Blacks']},
        '颜色分级': {k: params.get(k) for k in [
            'SplitToningShadowHue','SplitToningHighlightHue',
            'SplitToningShadowSaturation','SplitToningHighlightSaturation'
        ]},
    }

    prompt = f"""这是一张图片的风格分析结果：

场景信息：{json.dumps(scene_info, ensure_ascii=False)}

算法初步生成的Lightroom参数：
{json.dumps(key_params, ensure_ascii=False, indent=2)}

重要说明：以上参数是"相对调整量"，将被应用到一张中性/未调色的源图上，而不是当前这张参考图。
参考图已经代表目标风格，参数的作用是让中性源图变成参考图的样子。
因此：如果参考图橙色饱和度高，SaturationAdjustmentOrange应为正值（提升）；如果偏低，应为负值。
请不要因为参考图本身看起来"已经很饱和"就把正值纠正为0或负值。

请：
1. 检查色相（Hue）参数是否方向正确（如参考图偏绿，绿色HueAdjustment应向黄/蓝方向偏移）
2. 检查明度（Luminance）参数是否符合参考图的明暗风格
3. 补充算法可能遗漏的风格特征（如胶片感颗粒、特定色调偏移）
4. 不要修改饱和度（Saturation）参数，除非方向明显错误

以JSON格式返回：
{{
  "corrections": {{"参数名": 修正后的值}},
  "grain_amount": 0-50的数值,
  "style_summary": "一句话总结风格",
  "param_notes": "简短的参数说明"
}}

只返回JSON。"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text",  "text": prompt}
                ]
            }]
        )
        text       = response.content[0].text.strip().replace('```json', '').replace('```', '').strip()
        correction = json.loads(text)

        if 'corrections' in correction:
            for k, v in correction['corrections'].items():
                if k in params:
                    params[k] = _clamp_correction(k, v)

        params['GrainAmount']    = correction.get('grain_amount', 0)
        params['_style_summary'] = correction.get('style_summary', '')
        params['_param_notes']   = correction.get('param_notes', '')

    except Exception:
        params['GrainAmount']    = 5 if scene_info.get('grain_visible') else 0
        params['_style_summary'] = ''
        params['_param_notes']   = ''

    return params


def _apply_scene_rules(params: dict, scene_type: str) -> dict:
    """按场景类型应用参数约束"""
    rules  = SCENE_RULES.get(scene_type, SCENE_RULES['other'])
    result = dict(params)

    for key, rule in rules.items():
        if key in ('max_sat_global', 'needs_portrait_mask'):
            continue
        if key in result and callable(rule):
            result[key] = rule(result[key])

    max_sat = rules.get('max_sat_global', 70)
    for k in list(result.keys()):
        if k.startswith('SaturationAdjustment'):
            result[k] = clamp(result.get(k, 0), -100, max_sat)

    result['_needs_portrait_mask'] = rules.get('needs_portrait_mask', False)
    return result


def _compute_confidence(params: dict, scene_info: dict, ref_scene_type: str) -> dict:
    """计算各模块置信度"""
    wb_conf       = {'low': 0.5, 'very_low': 0.3}.get(params.get('wb_confidence', 'low'), 0.5)
    user_specified = scene_info.get('user_specified', False)

    return {
        'overall':           round(0.82 if user_specified else (0.75 if scene_info.get('api_success') else 0.55), 2),
        'hsl':               0.85,
        'color_grading':     0.78 if params.get('is_complementary_grading') else 0.58,
        'tone_curve':        0.65,
        'white_balance':     wb_conf,
        'scene_recognition': 1.0 if user_specified else (0.90 if scene_info.get('api_success') else 0.0),
    }


def _generate_report(
    scene_info:    dict,
    params:        dict,
    confidence:    dict,
    conflict_info: dict,
    src_scene:     Optional[str],
) -> dict:
    """生成分析报告"""
    warnings = []

    # 场景冲突提示（最优先）
    if conflict_info.get('has_conflict'):
        warnings.append(conflict_info['message'])

    # 人像蒙版建议
    if params.get('_needs_portrait_mask'):
        warnings.append("ℹ️ 人像场景：色彩风格已尽可能还原参考图效果，亮度和白平衡建议根据原片实际拍摄条件微调。如需精细调色可在LR中添加人像蒙版单独处理肤色。")

    # 白平衡低置信度
    if confidence['white_balance'] < 0.5:
        warnings.append("⚠️ 白平衡仅供参考方向，建议根据实际光源手动校准。")

    # AI不可用
    if not scene_info.get('api_success') and not scene_info.get('user_specified'):
        warnings.append("⚠️ AI场景识别不可用，参数基于纯算法分析。")

    # 颗粒感
    if params.get('GrainAmount', 0) > 0:
        warnings.append(f"ℹ️ 检测到胶片颗粒感，已添加颗粒量：{params['GrainAmount']}。")

    # 用户指定场景
    if scene_info.get('user_specified'):
        warnings.append(f"ℹ️ 场景类型由用户指定：{SCENE_LABELS.get(scene_info['scene_type'], scene_info['scene_type'])}。")

    return {
        'style_summary':   params.get('_style_summary', scene_info.get('estimated_style', '')),
        'scene_type':      scene_info.get('scene_type', 'other'),
        'scene_label':     SCENE_LABELS.get(scene_info.get('scene_type', 'other'), ''),
        'src_scene_label': SCENE_LABELS.get(src_scene, '') if src_scene else '',
        'style_keywords':  scene_info.get('style_keywords', []),
        'lighting':        scene_info.get('lighting', ''),
        'warnings':        warnings,
        'confidence':      confidence,
        'param_notes':     params.get('_param_notes', ''),
    }


def _compress_for_api(image_bytes: bytes, max_size: int = 1200) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode('utf-8')
    except Exception:
        return base64.standard_b64encode(image_bytes[:500000]).decode('utf-8')


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


_CORRECTION_LIMITS: dict = {
    'Exposure':   (-2.0, 2.0),
    'Contrast':   (-80,  80),
    'Highlights': (-100, 100),
    'Shadows':    (-100, 100),
    'Whites':     (-100, 100),
    'Blacks':     (-100, 100),
    'Clarity':    (-100, 100),
    'Vibrance':   (-60,  60),
    'Saturation': (-60,  60),
}

def _clamp_correction(key: str, value):
    """对 Claude 返回的校正值施加安全上限"""
    lo, hi = _CORRECTION_LIMITS.get(key, (-100, 100))
    try:
        v = float(value)
        if isinstance(lo, float):
            return round(max(lo, min(hi, v)), 2)
        return int(max(lo, min(hi, round(v))))
    except (TypeError, ValueError):
        return value
