"""一次性把 CR3 RAW 批量转换为 JPG（512×512），后续训练数据生成不再需要 darktable"""
import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def convert_one(src_path: str, out_dir: str, size: int = 512) -> bool:
    """转换单张 RAW → JPG"""
    src = Path(src_path)
    out = Path(out_dir) / (src.stem + '.jpg')
    if out.exists():
        return True
    try:
        result = subprocess.run(
            ['darktable-cli', str(src), str(out),
             '--width', str(size), '--height', str(size),
             '--hq', '1', '--apply-custom-presets', '0'],
            capture_output=True, timeout=60,
        )
        return result.returncode == 0 and out.exists()
    except Exception as e:
        print(f'  错误 {src.name}: {e}')
        return False


def main():
    src_dir = './photos'
    out_dir = './photos_jpg'
    os.makedirs(out_dir, exist_ok=True)

    files = list(Path(src_dir).glob('*.CR3'))
    print(f'找到 {len(files)} 张 RAW 照片')
    print(f'输出目录: {out_dir}')
    print(f'目标尺寸: 512×512\n')

    n_done = n_fail = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(convert_one, str(f), out_dir): f for f in files}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='转换中'):
            if fut.result():
                n_done += 1
            else:
                n_fail += 1

    print(f'\n✓ 完成: {n_done} 成功, {n_fail} 失败')


if __name__ == '__main__':
    main()
