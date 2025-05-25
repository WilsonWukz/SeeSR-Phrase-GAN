import os
from PIL import Image

def resize_images(input_dir: str, output_dir: str, max_size: int = 256):
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        # 只处理包含 'LR' 且为图像格式的文件
        if 'LR' not in fname:
            continue
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        src_path = os.path.join(input_dir, fname)
        try:
            with Image.open(src_path) as img:
                w, h = img.size
                scale = min(max_size / w, max_size / h, 1.0)
                new_w = int(w * scale)
                new_h = int(h * scale)
                if scale < 1.0:
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                dst_path = os.path.join(output_dir, fname)
                img.save(dst_path)
                print(f"Resized {fname}: {w}x{h} -> {new_w}x{new_h}")
        except Exception as e:
            print(f"Failed to process {fname}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resize images containing 'LR' in filename from multiple folders")

    parser.add_argument(
        "--base_dir", type=str,
        default=r"E:\Sydney_study\5329\A2\RealSR (ICCV2019)\Canon\Test",
        help="Base directory containing folders like 2, 3, 4"
    )
    parser.add_argument(
        "--output_base_dir", type=str,
        default=r"C:\Users\C\SeeSR\preset\datasets\test_datasets",
        help="Where to save resized images"
    )
    parser.add_argument(
        "--subfolders", type=str, nargs='+', default=['4'],
        help="Subfolders to process"
    )
    parser.add_argument(
        "--max_size", type=int, default=256,
        help="Maximum width/height after resizing"
    )
    args = parser.parse_args()

    for folder in args.subfolders:
        input_dir = os.path.normpath(os.path.join(args.base_dir, folder))
        output_dir = os.path.normpath(os.path.join(args.output_base_dir, folder, "LR_resized"))
        print(f"\nProcessing: {input_dir} → {output_dir}")
        if not os.path.exists(input_dir):
            print(f"[跳过] 输入路径不存在: {input_dir}")
            continue
        resize_images(input_dir, output_dir, args.max_size)
