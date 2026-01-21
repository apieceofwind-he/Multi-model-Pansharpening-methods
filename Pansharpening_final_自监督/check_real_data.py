# check_real_data.py
import os
from pathlib import Path


def check_real_data_structure():
    data_root = './data'

    print("检查真实测试数据目录结构...")
    print("=" * 60)

    # 检查根目录
    root_path = Path(data_root)
    if not root_path.exists():
        print(f"错误: 数据根目录不存在: {root_path}")
        return False

    # 列出根目录下的所有内容
    print(f"数据根目录内容 ({root_path}):")
    for item in root_path.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
        else:
            print(f"  📄 {item.name}")

    print("\n检查真实测试图片目录...")
    real_data_path = root_path / '真实测试图片'

    if not real_data_path.exists():
        print(f"错误: 真实测试图片目录不存在: {real_data_path}")

        # 检查其他可能的目录名
        possible_names = ['real_data', 'real_test', '真实测试', 'test_real']
        for name in possible_names:
            alt_path = root_path / name
            if alt_path.exists():
                print(f"找到替代目录: {alt_path}")
                real_data_path = alt_path
                break

        if not real_data_path.exists():
            print("请创建正确的目录结构:")
            print(f"{root_path}/")
            print("  ├── train_data/train/")
            print("  ├── test_data/test/")
            print("  └── 真实测试图片/")
            print("      ├── MS_up_800/")
            print("      └── PAN_cut_800/")
            return False

    print(f"\n真实测试图片目录结构 ({real_data_path}):")
    for item in real_data_path.iterdir():
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # 列出子目录内容
            for subitem in item.iterdir()[:3]:  # 只显示前3个
                if subitem.is_file():
                    print(f"    📄 {subitem.name}")
            if len(list(item.iterdir())) > 3:
                print(f"    ... 还有 {len(list(item.iterdir())) - 3} 个文件")
        else:
            print(f"  📄 {item.name}")

    # 检查MS_up_800和PAN_cut_800
    ms_dir = real_data_path / 'MS_up_800'
    pan_dir = real_data_path / 'PAN_cut_800'

    if ms_dir.exists():
        ms_files = list(ms_dir.glob('*.tif'))
        print(f"\nMS_up_800目录: {len(ms_files)} 个TIFF文件")
        for f in ms_files[:3]:
            print(f"  📄 {f.name}")
        if len(ms_files) > 3:
            print(f"  ... 还有 {len(ms_files) - 3} 个文件")
    else:
        print(f"\n警告: MS_up_800目录不存在: {ms_dir}")

    if pan_dir.exists():
        pan_files = list(pan_dir.glob('*.tif'))
        print(f"\nPAN_cut_800目录: {len(pan_files)} 个TIFF文件")
        for f in pan_files[:3]:
            print(f"  📄 {f.name}")
        if len(pan_files) > 3:
            print(f"  ... 还有 {len(pan_files) - 3} 个文件")
    else:
        print(f"\n警告: PAN_cut_800目录不存在: {pan_dir}")

    return True


if __name__ == "__main__":
    check_real_data_structure()