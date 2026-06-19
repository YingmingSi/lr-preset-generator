#!/bin/bash
# 检查数据生成 + 自动启动训练
# 用法: ./check_status.sh           # 只检查状态
#       ./check_status.sh --train   # 准备就绪后自动启动训练
#       ./check_status.sh --watch   # 持续监控（每30秒刷新一次）

TARGET=5000
DATA_DIR="./data"
AUTO_TRAIN=false
WATCH=false

for arg in "$@"; do
    case $arg in
        --train) AUTO_TRAIN=true ;;
        --watch) WATCH=true ;;
    esac
done

cd "$(dirname "$0")"

check_status() {
    local count=$(find "$DATA_DIR" -name "*_params.json" 2>/dev/null | wc -l)
    local gen_pid=$(ps aux | grep "generate_dataset.py" | grep -v grep | awk '{print $2}' | head -1)
    local size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
    local pct=$((count * 100 / TARGET))

    echo "═══════════════════════════════════════════════════════"
    echo "📊 数据生成状态  $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════"
    printf "进度: %d/%d (%d%%)\n" "$count" "$TARGET" "$pct"
    echo "磁盘占用: ${size:-N/A}"

    # 进度条
    local bar_len=40
    local filled=$((pct * bar_len / 100))
    printf "["
    for ((i=0; i<filled; i++)); do printf "█"; done
    for ((i=filled; i<bar_len; i++)); do printf "░"; done
    printf "] %d%%\n" "$pct"

    if [ -n "$gen_pid" ]; then
        echo "状态: ⏳ 生成中（PID: $gen_pid）"
        # 估算剩余时间（基于过去 60 秒的速度）
        if [ -f "/tmp/data_gen_last_check" ]; then
            local last_count=$(cat /tmp/data_gen_last_check 2>/dev/null || echo 0)
            local last_time=$(stat -c %Y /tmp/data_gen_last_check 2>/dev/null || echo 0)
            local now=$(date +%s)
            local delta_count=$((count - last_count))
            local delta_time=$((now - last_time))
            if [ "$delta_count" -gt 0 ] && [ "$delta_time" -gt 0 ]; then
                local rate=$((delta_count * 60 / delta_time))  # per minute
                local remaining=$(( (TARGET - count) / (rate + 1) ))
                echo "速率: ~$rate 对/分钟  |  剩余约 $remaining 分钟"
            fi
        fi
        echo "$count" > /tmp/data_gen_last_check
        return 1  # 还在进行
    else
        if [ "$count" -ge "$TARGET" ]; then
            echo "状态: ✅ 数据已就绪！"
            return 0
        elif [ "$count" -gt 0 ]; then
            echo "状态: ⚠️  生成进程已停止，但只完成 $count/$TARGET"
            echo "    可以用现有数据训练，或重新运行生成"
            return 0
        else
            echo "状态: ❌ 未启动数据生成"
            return 2
        fi
    fi
}

run_training() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "🚀 开始训练 CNN 模型"
    echo "═══════════════════════════════════════════════════════"
    echo "配置: 200 epochs, batch=32, lr=0.001, device=cuda"
    echo ""
    echo "提示: 训练过程中可在另一个终端查看 TensorBoard："
    echo "  python -m tensorboard.main --logdir=./checkpoints/logs"
    echo "  浏览器访问: http://localhost:6006"
    echo ""
    rm -rf checkpoints
    python train.py \
        --data-dir ./data \
        --epochs 200 \
        --batch-size 32 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --backbone resnet18 \
        --output-dir ./checkpoints \
        --device cuda \
        --num-workers 4 \
        --seed 42
}

if [ "$WATCH" = "true" ]; then
    # 持续监控
    while true; do
        clear
        check_status
        status=$?
        if [ $status -eq 0 ]; then
            echo ""
            echo "✅ 数据已就绪，停止监控"
            if [ "$AUTO_TRAIN" = "true" ]; then
                run_training
            else
                echo "💡 运行训练: ./check_status.sh --train"
            fi
            break
        fi
        echo ""
        echo "(按 Ctrl+C 退出监控，30 秒后刷新...)"
        sleep 30
    done
else
    # 单次检查
    check_status
    status=$?
    if [ $status -eq 0 ] && [ "$AUTO_TRAIN" = "true" ]; then
        run_training
    elif [ $status -ne 0 ]; then
        echo ""
        echo "💡 提示："
        echo "  - 持续监控并自动训练: ./check_status.sh --watch --train"
        echo "  - 持续监控（不训练）: ./check_status.sh --watch"
        echo "  - 单次检查: ./check_status.sh"
    fi
fi
