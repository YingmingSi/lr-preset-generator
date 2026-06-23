"""
把 PyTorch CNN 模型导出为 ONNX 格式

ONNX runtime 比 torch 轻 3-4 倍（50MB vs 200MB），适合内存受限的生产部署。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from cnn_model import ParamPredictor


def export_model(pt_path: str, onnx_path: str):
    """
    导出 PyTorch 模型为 ONNX

    Args:
        pt_path: PyTorch checkpoint 路径
        onnx_path: 输出 ONNX 路径
    """
    print(f"加载模型: {pt_path}")
    device = torch.device('cpu')
    model = ParamPredictor('simple_color', pretrained=False)
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 创建 dummy 输入（batch=1, 384x384 RGB）
    dummy_src = torch.rand(1, 3, 384, 384)
    dummy_ref = torch.rand(1, 3, 384, 384)

    print(f"导出 ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_src, dummy_ref),
        onnx_path,
        input_names=['src', 'ref'],
        output_names=['params'],
        dynamic_axes={
            'src':    {0: 'batch'},
            'ref':    {0: 'batch'},
            'params': {0: 'batch'},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # 文件大小
    pt_size = os.path.getsize(pt_path) / 1024 / 1024
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\n📦 文件大小:")
    print(f"  PyTorch: {pt_size:.1f} MB")
    print(f"  ONNX:    {onnx_size:.1f} MB")

    # 验证：PyTorch 和 ONNX 输出一致
    print(f"\n🔬 验证输出一致性...")
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    # 用真实数据测试
    test_src = torch.rand(1, 3, 384, 384)
    test_ref = torch.rand(1, 3, 384, 384)

    with torch.no_grad():
        pt_out = model(test_src, test_ref).numpy()

    onnx_out = sess.run(
        None,
        {'src': test_src.numpy(), 'ref': test_ref.numpy()}
    )[0]

    diff = np.abs(pt_out - onnx_out).max()
    print(f"  PyTorch 输出: {pt_out.flatten()[:5]}")
    print(f"  ONNX 输出:    {onnx_out.flatten()[:5]}")
    print(f"  最大差异: {diff:.6f}")

    if diff < 1e-4:
        print(f"\n✅ ONNX 导出成功且输出一致！")
    else:
        print(f"\n⚠ 差异较大，请检查 ONNX 导出")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt', default='../backend/models/param_predictor.pt',
                        help='PyTorch checkpoint')
    parser.add_argument('--onnx', default='../backend/models/param_predictor.onnx',
                        help='输出 ONNX 路径')
    args = parser.parse_args()

    export_model(args.pt, args.onnx)
