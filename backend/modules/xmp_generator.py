"""
XMP文件生成器
将分析参数映射到Lightroom Classic格式的XMP预设文件
"""

import uuid
import os
from typing import Optional
from modules.luminance_analyzer import format_tone_curve_xml


DEFAULT_PARAMS = {
    'Temperature': 5500,
    'Tint': 0,
    'Exposure': 0.0,
    'Contrast': 0,
    'Highlights': 0,
    'Shadows': 0,
    'Whites': 0,
    'Blacks': 0,
    'Clarity': 0,
    'Texture': 0,
    'Dehaze': 0,
    'Vibrance': 0,
    'Saturation': 0,
    'ParametricShadows': 0,
    'ParametricDarks': 0,
    'ParametricLights': 0,
    'ParametricHighlights': 0,
    'LuminanceSmoothing': 0,
    'ColorNoiseReduction': 25,
    'HueAdjustmentRed': 0,
    'HueAdjustmentOrange': 0,
    'HueAdjustmentYellow': 0,
    'HueAdjustmentGreen': 0,
    'HueAdjustmentAqua': 0,
    'HueAdjustmentBlue': 0,
    'HueAdjustmentPurple': 0,
    'HueAdjustmentMagenta': 0,
    'SaturationAdjustmentRed': 0,
    'SaturationAdjustmentOrange': 0,
    'SaturationAdjustmentYellow': 0,
    'SaturationAdjustmentGreen': 0,
    'SaturationAdjustmentAqua': 0,
    'SaturationAdjustmentBlue': 0,
    'SaturationAdjustmentPurple': 0,
    'SaturationAdjustmentMagenta': 0,
    'LuminanceAdjustmentRed': 0,
    'LuminanceAdjustmentOrange': 0,
    'LuminanceAdjustmentYellow': 0,
    'LuminanceAdjustmentGreen': 0,
    'LuminanceAdjustmentAqua': 0,
    'LuminanceAdjustmentBlue': 0,
    'LuminanceAdjustmentPurple': 0,
    'LuminanceAdjustmentMagenta': 0,
    'SplitToningShadowHue': 0,
    'SplitToningShadowSaturation': 0,
    'SplitToningHighlightHue': 0,
    'SplitToningHighlightSaturation': 0,
    'SplitToningBalance': 0,
    'ColorGradeMidtoneHue': 0,
    'ColorGradeMidtoneSat': 0,
    'ColorGradeShadowLum': 0,
    'ColorGradeHighlightLum': 0,
    'GrainAmount': 0,
}

NEUTRAL_CURVE = [(0, 0), (64, 64), (128, 128), (192, 192), (255, 255)]


def generate_xmp(
    luminance_params: dict,
    color_params: dict,
    scene_result: dict,
    preset_name: str = "AI生成预设",
    description: str = "",
) -> str:
    merged = dict(DEFAULT_PARAMS)
    merged.update(_extract_flat_params(luminance_params))
    merged.update(_extract_flat_params(color_params))
    merged.update(_extract_flat_params(scene_result.get('params', {})))

    tone_curve  = luminance_params.get('tone_curve',       NEUTRAL_CURVE)
    red_curve   = color_params.get('tone_curve_red',   NEUTRAL_CURVE)
    green_curve = color_params.get('tone_curve_green', NEUTRAL_CURVE)
    blue_curve  = color_params.get('tone_curve_blue',  NEUTRAL_CURVE)

    report        = scene_result.get('report', {})
    style_summary = report.get('style_summary', '')
    auto_desc     = f"{style_summary} | AI分析生成" if style_summary else "AI分析生成"
    if description:
        auto_desc = description

    template_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'templates', 'preset_template.xmp'
    )
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    xmp = template
    for key, value in merged.items():
        placeholder = '{' + key + '}'
        if placeholder in xmp:
            xmp = xmp.replace(placeholder, str(value))

    xmp = xmp.replace('{ToneCurvePV2012}',      format_tone_curve_xml(tone_curve))
    xmp = xmp.replace('{ToneCurvePV2012Red}',   format_tone_curve_xml(red_curve))
    xmp = xmp.replace('{ToneCurvePV2012Green}', format_tone_curve_xml(green_curve))
    xmp = xmp.replace('{ToneCurvePV2012Blue}',  format_tone_curve_xml(blue_curve))
    xmp = xmp.replace('{UUID}',        str(uuid.uuid4()).upper())
    xmp = xmp.replace('{PresetName}',  preset_name)
    xmp = xmp.replace('{Description}', auto_desc)

    return xmp


def _extract_flat_params(params: dict) -> dict:
    result = {}
    for k, v in params.items():
        if k.startswith('_'):
            continue
        if k in ('tone_curve', 'tone_curve_red', 'tone_curve_green', 'tone_curve_blue'):
            continue
        if k in ('wb_confidence', 'is_complementary_grading'):
            continue
        if k in DEFAULT_PARAMS:
            result[k] = v
    return result


def params_summary(luminance_params: dict, color_params: dict, scene_result: dict) -> dict:
    """返回人类可读的参数摘要，用于前端展示"""
    p = scene_result.get('params', {})

    return {
        '基础调整': {
            '曝光':     luminance_params.get('Exposure', 0),
            '对比度':   luminance_params.get('Contrast', 0),
            '高光':     luminance_params.get('Highlights', 0),
            '阴影':     luminance_params.get('Shadows', 0),
            '白色':     luminance_params.get('Whites', 0),
            '黑色':     luminance_params.get('Blacks', 0),
            '清晰度':   luminance_params.get('Clarity', 0),
            '自然饱和度': color_params.get('Vibrance', 0),
        },
        '白平衡': {
            '色温':   color_params.get('Temperature', 5500),
            '色调':   color_params.get('Tint', 0),
            '置信度': '低（建议根据原片手动校准）',
        },
        'HSL色相偏移': {
            k.replace('HueAdjustment', ''): v
            for k, v in p.items()
            if k.startswith('HueAdjustment') and v != 0
        },
        'HSL饱和度': {
            k.replace('SaturationAdjustment', ''): v
            for k, v in p.items()
            if k.startswith('SaturationAdjustment') and v != 0
        },
        'HSL明度': {
            k.replace('LuminanceAdjustment', ''): v
            for k, v in p.items()
            if k.startswith('LuminanceAdjustment') and v != 0
        },
        '颜色分级': {
            '阴影色相':   p.get('SplitToningShadowHue', 0),
            '阴影饱和度': p.get('SplitToningShadowSaturation', 0),
            '高光色相':   p.get('SplitToningHighlightHue', 0),
            '高光饱和度': p.get('SplitToningHighlightSaturation', 0),
        },
    }
