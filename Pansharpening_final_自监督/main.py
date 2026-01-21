
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pan-GAN 修复版训练和测试')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'test', 'real_test', 'evaluate'],
                        help='运行模式: train, test, real_test, evaluate')

    args = parser.parse_args()

    # 创建目录
    os.makedirs(CONFIG['data_root'], exist_ok=True)
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['result_dir'], exist_ok=True)

    if args.mode == 'test':
        test_model('test')

    elif args.mode == 'real_test':
        # 设置批次大小为1，因为真实图像较大
        CONFIG['batch_size'] = 1
        test_model('real')

    elif args.mode == 'evaluate':
        evaluate_test_results()

    else:
        print(f"训练模式暂未实现，请先使用测试模式")
        print("可用模式: test, real_test, evaluate")