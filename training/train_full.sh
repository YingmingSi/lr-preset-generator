#!/bin/bash

#################################################################
# LR Preset Generator - 完整训练流程自动脚本
#
# 用法：
#   ./train_full.sh              # 完整流程 (4 步)
#   ./train_full.sh --step 1     # 仅生成测试数据
#   ./train_full.sh --step 2     # 仅生成正式数据
#   ./train_full.sh --step 3     # 仅快速验证训练
#   ./train_full.sh --step 4     # 仅正式训练
#   ./train_full.sh --device cpu # 使用 CPU 训练
#
#################################################################

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 脚本参数
DEVICE="cuda"  # 默认使用 GPU
STEP=0         # 0 = 所有步骤，1-4 = 指定步骤
DATA_WORKERS=8 # 数据生成工作进程
BATCH_SIZE=32  # GPU 批大小
CPU_BATCH_SIZE=8  # CPU 批大小

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --workers)
            DATA_WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
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

    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python3 未安装${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ Python3: $(python3 --version)${NC}"
    fi

    # 检查 PyTorch
    if ! python3 -c "import torch" 2>/dev/null; then
        echo -e "${RED}❌ PyTorch 未安装${NC}"
        echo -e "${YELLOW}   运行: pip install torch torchvision${NC}"
        missing=1
    else
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}✓ PyTorch: $TORCH_VERSION${NC}"
    fi

    # 检查 Darktable
    if ! command -v darktable-cli &> /dev/null; then
        echo -e "${RED}❌ Darktable 未安装${NC}"
        echo -e "${YELLOW}   运行: sudo apt install darktable (Linux)${NC}"
        echo -e "${YELLOW}   或:   brew install darktable (macOS)${NC}"
        missing=1
    else
        echo -e "${GREEN}✓ Darktable: $(darktable-cli --version 2>/dev/null | head -1)${NC}"
    fi

    # 检查 TensorBoard
    if ! python3 -c "import tensorboard" 2>/dev/null; then
        echo -e "${YELLOW}⚠ TensorBoard 未安装（可选）${NC}"
        echo -e "${YELLOW}   运行: pip install tensorboard${NC}"
    else
        echo -e "${GREEN}✓ TensorBoard 已安装${NC}"
    fi

    if [ $missing -eq 1 ]; then
        echo -e "${RED}❌ 缺少必要依赖，请先安装${NC}"
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

# 步骤 1: 生成测试数据
step1_generate_test_data() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}📊 步骤 1️⃣: 生成小规模测试数据 (100 对)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ -d "./data_test" ]; then
        echo -e "${YELLOW}⚠ data_test 目录已存在，跳过生成${NC}"
        TEST_COUNT=$(find ./data_test -name "*_params.json" | wc -l)
        echo -e "${GREEN}✓ 已有 $TEST_COUNT 对测试数据${NC}\n"
        return 0
    fi

    echo "生成 100 对测试数据（用于快速验证）..."
    python generate_dataset.py \
        --src-dir ./photos \
        --out-dir ./data_test \
        --n-pairs 100 \
        --n-workers $DATA_WORKERS \
        --img-size 384

    TEST_COUNT=$(find ./data_test -name "*_params.json" | wc -l)
    if [ $TEST_COUNT -lt 100 ]; then
        echo -e "${RED}❌ 生成失败：仅生成了 $TEST_COUNT 对（预期 100 对）${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ 测试数据生成完成！${NC}"
    echo -e "${GREEN}   生成了 $TEST_COUNT 对数据${NC}"
    echo -e "${GREEN}   磁盘占用: $(du -sh ./data_test | cut -f1)${NC}\n"
}

# 步骤 2: 生成正式数据
step2_generate_train_data() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}📊 步骤 2️⃣: 生成正式训练数据 (5000 对)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    TRAIN_COUNT=$(find ./data -name "*_params.json" 2>/dev/null | wc -l)
    if [ $TRAIN_COUNT -ge 5000 ]; then
        echo -e "${YELLOW}⚠ 训练数据已存在，跳过生成${NC}"
        echo -e "${GREEN}✓ 已有 $TRAIN_COUNT 对训练数据${NC}"
        echo -e "${GREEN}   磁盘占用: $(du -sh ./data | cut -f1)${NC}\n"
        return 0
    fi

    echo "生成 5000 对训练数据..."
    echo -e "${YELLOW}⏳ 这可能需要 1-5 小时（取决于你的 CPU）${NC}"

    python generate_dataset.py \
        --src-dir ./photos \
        --out-dir ./data \
        --n-pairs 5000 \
        --n-workers $DATA_WORKERS \
        --img-size 384

    TRAIN_COUNT=$(find ./data -name "*_params.json" | wc -l)
    if [ $TRAIN_COUNT -lt 5000 ]; then
        echo -e "${RED}❌ 生成失败：仅生成了 $TRAIN_COUNT 对（预期 5000 对）${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ 训练数据生成完成！${NC}"
    echo -e "${GREEN}   生成了 $TRAIN_COUNT 对数据${NC}"
    echo -e "${GREEN}   磁盘占用: $(du -sh ./data | cut -f1)${NC}\n"
}

# 步骤 3: 快速验证训练
step3_quick_verify() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}🔬 步骤 3️⃣: 快速验证训练脚本 (20 epoch)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${YELLOW}用途: 验证代码无误，估算完整训练时间${NC}\n"

    if [ $DEVICE = "cuda" ]; then
        VERIFY_BATCH=16
    else
        VERIFY_BATCH=8
    fi

    python train.py \
        --data-dir ./data_test \
        --epochs 20 \
        --batch-size $VERIFY_BATCH \
        --lr 0.001 \
        --device $DEVICE \
        --output-dir ./checkpoints_test \
        --num-workers 2

    echo -e "${GREEN}✓ 快速验证完成！${NC}\n"
}

# 步骤 4: 正式训练
step4_full_training() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}🚀 步骤 4️⃣: 正式训练 CNN 模型 (150 epoch)${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ $DEVICE = "cuda" ]; then
        echo -e "${GREEN}📱 使用 GPU 训练${NC}"
        echo -e "${YELLOW}⏳ 预计耗时: 2-8 小时${NC}\n"
        TRAIN_BATCH=$BATCH_SIZE
        TRAIN_WORKERS=4
    else
        echo -e "${YELLOW}📱 使用 CPU 训练${NC}"
        echo -e "${RED}⏳ 预计耗时: 24-48 小时${NC}\n"
        TRAIN_BATCH=$CPU_BATCH_SIZE
        TRAIN_WORKERS=2
    fi

    # 启动 TensorBoard 提示
    echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}"
    echo -e "${BLUE}💡 监控训练进度（在另一个终端执行）:${NC}"
    echo -e "${YELLOW}   tensorboard --logdir=./checkpoints/logs${NC}"
    echo -e "${YELLOW}   然后访问: http://localhost:6006${NC}"
    echo -e "${BLUE}───────────────────────────────────────────────────────────${NC}\n"

    python train.py \
        --data-dir ./data \
        --epochs 150 \
        --batch-size $TRAIN_BATCH \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --backbone resnet18 \
        --output-dir ./checkpoints \
        --device $DEVICE \
        --num-workers $TRAIN_WORKERS \
        --seed 42

    echo -e "${GREEN}✓ 正式训练完成！${NC}\n"
}

# 显示最终结果
show_results() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}📈 训练结果${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if [ -f "./checkpoints/test_results.json" ]; then
        echo -e "${GREEN}✓ 测试结果文件已生成${NC}\n"

        # 提取关键指标
        R2=$(python3 -c "import json; d=json.load(open('./checkpoints/test_results.json')); print(d['test_metrics']['r2_mean'])" 2>/dev/null || echo "N/A")
        MAE=$(python3 -c "import json; d=json.load(open('./checkpoints/test_results.json')); print(d['test_metrics']['mae'])" 2>/dev/null || echo "N/A")
        LOSS=$(python3 -c "import json; d=json.load(open('./checkpoints/test_results.json')); print(d['test_metrics']['loss'])" 2>/dev/null || echo "N/A")

        echo "关键指标:"
        echo -e "  R² 得分:     $R2 (目标 > 0.80)"
        echo -e "  MAE:         $MAE"
        echo -e "  Loss:        $LOSS"
        echo ""

        if [ "$R2" != "N/A" ]; then
            R2_NUM=$(echo "$R2" | cut -d'.' -f1,2)
            if (( $(echo "$R2_NUM > 0.85" | bc -l) )); then
                echo -e "${GREEN}✓ 优秀！R² > 0.85${NC}"
            elif (( $(echo "$R2_NUM > 0.80" | bc -l) )); then
                echo -e "${GREEN}✓ 很好！R² > 0.80${NC}"
            else
                echo -e "${YELLOW}⚠ 可以，但有改进空间（R² < 0.80）${NC}"
            fi
        fi
        echo ""

        echo "完整结果查看:"
        echo -e "  ${YELLOW}cat ./checkpoints/test_results.json | python -m json.tool${NC}"
    else
        echo -e "${YELLOW}⚠ 还未生成测试结果${NC}"
    fi

    echo ""
}

# 显示最佳模型
show_best_model() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}🏆 最佳模型${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    if ls ./checkpoints/best_model_*.pt 1> /dev/null 2>&1; then
        echo "找到的模型文件:"
        ls -lh ./checkpoints/best_model_*.pt | awk '{print "  " $9 " (" $5 ")"}'
        echo ""

        BEST_MODEL=$(ls -t ./checkpoints/best_model_*.pt | head -1)
        echo -e "${GREEN}最新最佳模型:${NC}"
        echo -e "  ${YELLOW}$BEST_MODEL${NC}"
        echo ""
        echo "下一步："
        echo -e "  ${YELLOW}cp $BEST_MODEL ../backend/models/param_predictor.pt${NC}"
    else
        echo -e "${YELLOW}⚠ 还未找到模型文件${NC}"
    fi

    echo ""
}

# 显示后续步骤
show_next_steps() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}📋 后续步骤${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"

    echo "1. 复制模型到后端:"
    echo -e "   ${YELLOW}cp ./checkpoints/best_model_epoch*.pt ../backend/models/param_predictor.pt${NC}"
    echo ""

    echo "2. 在 backend/main.py 中集成 CNN（见 ACTION_PLAN.md）"
    echo ""

    echo "3. 启动后端测试:"
    echo -e "   ${YELLOW}cd ../backend${NC}"
    echo -e "   ${YELLOW}python main.py${NC}"
    echo ""

    echo "4. 测试 API:"
    echo -e "   ${YELLOW}curl http://localhost:8000/health${NC}"
    echo ""
}

# 主函数
main() {
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     LR Preset Generator - CNN 训练自动脚本              ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}\n"

    # 显示配置
    echo "配置:"
    echo "  设备: $DEVICE"
    echo "  数据生成 workers: $DATA_WORKERS"
    echo ""

    # 检查依赖和源图
    check_dependencies
    check_photos

    # 执行步骤
    if [ $STEP -eq 0 ] || [ $STEP -eq 1 ]; then
        step1_generate_test_data
    fi

    if [ $STEP -eq 0 ] || [ $STEP -eq 2 ]; then
        step2_generate_train_data
    fi

    if [ $STEP -eq 0 ] || [ $STEP -eq 3 ]; then
        step3_quick_verify
    fi

    if [ $STEP -eq 0 ] || [ $STEP -eq 4 ]; then
        step4_full_training
    fi

    # 显示结果
    echo ""
    show_results
    show_best_model
    show_next_steps

    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🎉 所有步骤完成！${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
}

# 错误处理
trap 'echo -e "${RED}❌ 脚本执行出错！${NC}"; exit 1' ERR

# 运行
main
