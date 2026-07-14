"""
图像加载与预处理模块
支持RAW格式（CR2/CR3/NEF/ARW/RAF/DNG）和JPG/PNG
输出统一的sRGB色彩空间图像用于后续分析
"""

import numpy as np
from PIL import Image, ImageCms
import io
import os
import tempfile

# RAW格式后缀
RAW_EXTENSIONS = {'.cr2', '.cr3', '.nef', '.arw', '.raf', '.dng', '.rw2', '.orf', '.pef'}


def load_image(file_bytes: bytes, filename: str) -> dict:
    """
    加载图像文件，返回标准化的图像数据字典

    Returns:
        {
            'rgb_uint8':    np.ndarray (H, W, 3) uint8  sRGB 用于色彩分析
            'rgb_float':    np.ndarray (H, W, 3) float  0-1  用于精确计算
            'is_raw':       bool
            'width':        int
            'height':       int
            'compression_suspected': bool  是否疑似平台压缩
        }
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext in RAW_EXTENSIONS:
        return _load_raw(file_bytes, filename)
    else:
        return _load_standard(file_bytes)


def _load_raw(file_bytes: bytes, filename: str) -> dict:
    """加载RAW文件，输出线性光数据转换后的sRGB"""
    try:
        import rawpy
    except ImportError:
        raise RuntimeError("rawpy未安装，无法处理RAW文件")

    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name

    try:
        with rawpy.imread(tmp_path) as raw:
            # 使用sRGB色彩空间输出，相机白平衡，16bit
            # 不设置no_auto_bright，使rawpy自动亮度校正与LR默认渲染一致
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB,
                output_bps=16,
            )
        # 转换为8bit
        rgb8 = (rgb16 / 256).astype(np.uint8)
        # 先降采样再转 float，避免全分辨率 float 占满内存（免费层 512MB）
        rgb8 = _resize8(rgb8)
        rgb_float = rgb8.astype(np.float32) / 255.0

        return {
            'rgb_uint8': rgb8,
            'rgb_float': rgb_float,
            'is_raw': True,
            'width': rgb8.shape[1],
            'height': rgb8.shape[0],
            'compression_suspected': False,
        }
    finally:
        os.unlink(tmp_path)


def _load_standard(file_bytes: bytes) -> dict:
    """加载JPG/PNG，检测压缩特征，转换到sRGB"""
    img = Image.open(io.BytesIO(file_bytes))

    # 检测色彩配置文件并转换到sRGB
    img = _convert_to_srgb(img)

    # 转为RGB（去除alpha通道）
    if img.mode != 'RGB':
        img = img.convert('RGB')

    rgb8 = np.array(img)
    compression_suspected = _detect_compression(rgb8, file_bytes)

    if compression_suspected:
        rgb8 = _compensate_compression(rgb8)

    # 先降采样再转 float，避免全分辨率 float 占满内存（免费层 512MB）
    rgb8 = _resize8(rgb8)
    rgb_float = rgb8.astype(np.float32) / 255.0

    return {
        'rgb_uint8': rgb8,
        'rgb_float': rgb_float,
        'is_raw': False,
        'width': rgb8.shape[1],
        'height': rgb8.shape[0],
        'compression_suspected': compression_suspected,
    }


def _convert_to_srgb(img: Image.Image) -> Image.Image:
    """将图像色彩配置文件转换到sRGB"""
    try:
        icc = img.info.get('icc_profile')
        if icc:
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            srgb_profile = ImageCms.createProfile('sRGB')
            img = ImageCms.profileToProfile(
                img.convert('RGB'),
                input_profile,
                srgb_profile,
                outputMode='RGB'
            )
    except Exception:
        pass
    return img


def _detect_compression(rgb8: np.ndarray, file_bytes: bytes) -> bool:
    """
    检测是否为社交媒体压缩图片
    特征：文件大小与像素数比值低，且存在块状伪影
    """
    h, w = rgb8.shape[:2]
    pixels = h * w
    file_size = len(file_bytes)
    bytes_per_pixel = file_size / pixels

    # 社交媒体图片通常 < 0.5 bytes/pixel
    if bytes_per_pixel < 0.5:
        return True

    # 检测JPEG块状伪影（8x8块边界的高频差异）
    if pixels > 100000:
        sample = rgb8[:min(h, 200), :min(w, 200), :]
        # 计算8像素间隔的梯度
        block_edges = np.abs(sample[8::8, :, :].astype(int) - sample[7:-1:8, :, :].astype(int))
        non_edges = np.abs(sample[1::2, :, :].astype(int) - sample[:-1:2, :, :].astype(int))
        if non_edges.mean() > 0 and block_edges.mean() / non_edges.mean() > 2.5:
            return True

    return False


def _compensate_compression(rgb8: np.ndarray) -> np.ndarray:
    """
    对疑似压缩图片进行补偿
    轻微提升饱和度和对比度，补偿压缩导致的损失
    """
    from PIL import ImageEnhance
    img = Image.fromarray(rgb8)
    # 饱和度补偿 +8%
    img = ImageEnhance.Color(img).enhance(1.08)
    # 对比度补偿 +5%
    img = ImageEnhance.Contrast(img).enhance(1.05)
    return np.array(img)


def _resize8(rgb8: np.ndarray, max_size: int = 512):
    """缩放 uint8 图像，最大边不超过 max_size，保持宽高比。
    本应用只把图喂给 384px CNN（LUT 烘焙不用图），512 足够且省内存。"""
    h, w = rgb8.shape[:2]
    if max(h, w) <= max_size:
        return rgb8
    scale = max_size / max(h, w)
    img = Image.fromarray(rgb8).resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return np.array(img)