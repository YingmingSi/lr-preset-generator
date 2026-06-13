#!/bin/bash

#################################################################
# LR Preset Generator - 完整训练脚本
#
# 自动执行完整的数据生成和模型训练流程
#
# 用法：
#   ./train_complete.sh              # 完整流程（5000对数据 + 150 epoch训练）
#   ./train_complete.sh --device cpu # 使用 CPU
#   ./train_complete.sh --n-pairs 10000 # 生成 10000 对数据
#
#################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
DEVICE="cuda"
N_PAIRS=5000
N_WORKERS=8
BATCH_SIZE=32
EPOCHS=150
CPU_BATCH_SIZE=8

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --n-pairs)
            N_PAIRS="$2"
            shift 2
            ;;
        --workers)
            N_WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}🔍 检查依赖${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    local missing=0

    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 未安装${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ Python3: $(python3 --version)${NC}"
    fi

    if ! python3 -c "import torch" 2>/dev/null; then
        echo -e "${RED}❌ PyTorch 未安装${NC}"
        missing=1
    else
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}✓ PyTorch: $TORCH_VERSION${NC}"
    fi

    if ! command -v darktable-cli &> /dev/null; then
        echo -e "${RED}❌ Darktable 未安装${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ Darktable 已安装${NC}"
    fi

    if [ $missing -eq 1 ]; then
        echo -e "${RED}❌ 缺少必要依赖${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ 所有依赖就绪${NC}\n"
}

# 检查源图
check_photos() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}🖼️  检查源图${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ ! -d "./photos" ]; then
        echo -e "${RED}❌ 照片目录不存在: ./photos${NC}"
        exit 1
    fi

    PHOTO_COUNT=$(find ./photos -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.raw" -o -iname "*.cr2" -o -iname "*.cr3" -o -iname "*.nef" -o -iname "*.arw" -o -iname "*.dng" \) | wc -l)

    if [ $PHOTO_COUNT -lt 100 ]; then
        echo -e "${RED}❌ 照片不足：$PHOTO_COUNT 张（需要至少 100 张）${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ 找到 $PHOTO_COUNT 张照片${NC}\n"
}

# 生成训练数据
generate_training_data() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}📊 生成 $N_PAIRS 对训练数据${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    EXISTING=$(find ./data -name "*_params.json" 2>/dev/null | wc -l)
    if [ $EXISTING -ge $N_PAIRS ]; then
        echo -e "${YELLOW}⚠ 训练数据已存在，跳过生成${NC}"
        echo -e "${GREEN}✓ 已有 $EXISTING 对训练数据${NC}"
        echo -e "${GREEN}   磁盘占用: $(du -sh ./data 2>/dev/null | cut -f1)${NC}\n"
        return 0
    fi

    echo "生成 $N_PAIRS 对训练数据..."
    echo -e "${YELLOW}⏳ 这可能需要 1-10 小时（取决于 CPU 和数据量）${NC}\n"

    python generate_dataset.py \
        --src-dir ./photos \
        --out-dir ./data \
        --n-pairs $N_PAIRS \
        --n-workers $N_WORKERS \
        --img-size 384

    GENERATED=$(find ./data -name "*_params.json" | wc -l)
    if [ $GENERATED -lt $N_PAIRS ]; then
        echo -e "${YELLOW}⚠ 仅生成了 $GENERATED 对（可能是数据生成被中断）${NC}"
    fi

    echo -e "${GREEN}✓ 训练数据生成完成！${NC}"
    echo -e "${GREEN}   生成了 $GENERATED 对数据${NC}"
    echo -e "${GREEN}   磁盘占用: $(du -sh ./data | cut -f1)${NC}\n"
}

# 完整训练
full_training() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}🚀 完整 CNN 模型训练 ($EPOCHS epoch)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ $DEVICE = "cuda" ]; then
        echo -e "${GREEN}📱 使用 GPU 训练${NC}"
        echo -e "${YELLOW}⏳ 预计耗时: 2-8 小时${NC}\n"
        TRAIN_BATCH=$BATCH_SIZE
        TRAIN_WORKERS=4
    else
        echo -e "${YELLOW}📱 使用 CPU 训练${NC}"
        echo -e "${RED}⏳ 预计耗时: 20-48 小时${NC}\n"
        TRAIN_BATCH=$CPU_BATCH_SIZE
        TRAIN_WORKERS=2
    fi

    # 启动 TensorBoard 提示
    echo -e "${BLUE}───────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}💡 监控训练进度（在另一个终端执行）:${NC}"
    echo -e "${YELLOW}   python -m tensorboard.main --logdir=./checkpoints/logs${NC}"
    echo -e "${YELLOW}   然后访问: http://localhost:6006${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────${NC}\n"

    python train.py \
        --data-dir ./data \
        --epochs $EPOCHS \
        --batch-size $TRAIN_BATCH \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --backbone resnet18 \
        --output-dir ./checkpoints \
        --device $DEVICE \
        --num-workers $TRAIN_WORKERS \
        --seed 42

    echo -e "${GREEN}✓ 完整训练完成！${NC}\n"
}

# 显示最终结果
show_results() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}📈 最终结果${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ -f "./checkpoints/test_results.json" ]; then
        echo -e "${GREEN}✓ 测试结果文件已生成${NC}\n"

        python3 << 'EOF'
import json
try:
    with open('./checkpoints/test_results.json') as f:
        data = json.load(f)
    metrics = data['test_metrics']
    r2 = metrics['r2_mean']
    mae = metrics['mae']
    loss = metrics['loss']

    print("关键指标:")
    print(f"  R² 得分:     {r2:.4f} (目标 > 0.80)")
    print(f"  MAE:         {mae:.4f}")
    print(f"  Loss:        {loss:.4f}")
    print()

    if r2 > 0.85:
        print("✅ 优秀！R² > 0.85，可以部署！")
    elif r2 > 0.80:
        print("✅ 很好！R² > 0.80，精度不错")
    elif r2 > 0.70:
        print("🟡 可以，但有改进空间（R² > 0.70）")
    else:
        print("❌ 需要改进（考虑增加数据或调整超参数）")
except Exception as e:
    print(f"读取结果失败: {e}")
EOF
    else
        echo -e "${YELLOW}⚠ 还未生成测试结果${NC}"
    fi

    echo ""
}

# 显示后续步骤
show_next_steps() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}📋 后续步骤${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    echo "1. 复制最佳模型到后端:"
    echo -e "   ${YELLOW}cp ./checkpoints/best_model_epoch*.pt ../backend/models/param_predictor.pt${NC}"
    echo ""

    echo "2. 在 backend/main.py 中集成 CNN（见 ACTION_PLAN.md）"
    echo ""

    echo "3. 启动后端测试:"
    echo -e "   ${YELLOW}cd ../backend && python main.py${NC}"
    echo ""

    echo "4. 测试 API:"
    echo -e "   ${YELLOW}curl http://localhost:8000/health${NC}"
    echo ""
}

# 主函数
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     LR Preset Generator - 完整训练流程                    ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}\n"

    echo "配置:"
    echo "  设备: $DEVICE"
    echo "  数据对数: $N_PAIRS"
    echo "  训练 Epochs: $EPOCHS"
    echo "  数据生成 Workers: $N_WORKERS"
    echo ""

    check_dependencies
    check_photos
    generate_training_data
    full_training
    show_results
    show_next_steps

    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 完全训练流程完成！${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
}

# 错误处理
trap 'echo -e "${RED}❌ 脚本执行出错！${NC}"; exit 1' ERR

# 运行
main
