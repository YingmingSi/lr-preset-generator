#!/bin/bash
# 重启后续跑：数据生成 + 训练
#
# 用法：
#   ./resume_pipeline.sh           # 自动检测当前进度，继续
#   ./resume_pipeline.sh --status  # 只看状态，不启动

set -e
cd "$(dirname "$0")"

TARGET=26280

show_status() {
    echo "═══════════════════════════════════════════════════════"
    echo "📊 流水线状态  $(date '+%H:%M:%S')"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    # 数据
    DATA_COUNT=$(find ./data -name "*_params.json" 2>/dev/null | wc -l)
    DATA_PCT=$((DATA_COUNT * 100 / TARGET))
    GEN_PID=$(pgrep -f "generate_dataset.py" | head -1)
    if [ -n "$GEN_PID" ]; then
        echo "🔄 数据生成中 (PID $GEN_PID): $DATA_COUNT/$TARGET ($DATA_PCT%)"
    elif [ "$DATA_COUNT" -ge "$TARGET" ]; then
        echo "✅ 数据已就绪: $DATA_COUNT 对"
    else
        echo "⏸  数据生成已停止: $DATA_COUNT/$TARGET ($DATA_PCT%) - 需继续"
    fi

    # 训练
    LATEST_CKPT=$(ls -t checkpoints/best_model_*.pt 2>/dev/null | head -1)
    TRAIN_PID=$(pgrep -f "train.py" | head -1)
    if [ -n "$TRAIN_PID" ]; then
        echo "🔄 训练中 (PID $TRAIN_PID)"
        if [ -n "$LATEST_CKPT" ]; then
            BEST_EPOCH=$(basename "$LATEST_CKPT" | grep -oP "epoch\d+" | grep -oP "\d+")
            BEST_R2=$(basename "$LATEST_CKPT" | grep -oP "r2[-\d.]+")
            echo "   最新: epoch $BEST_EPOCH, $BEST_R2"
        fi
    elif [ -n "$LATEST_CKPT" ]; then
        BEST_EPOCH=$(basename "$LATEST_CKPT" | grep -oP "epoch\d+" | grep -oP "\d+")
        BEST_R2=$(basename "$LATEST_CKPT" | grep -oP "r2[-\d.]+")
        # 看是否已完成测试
        if [ -f checkpoints/test_results.json ]; then
            echo "✅ 训练完成: epoch $BEST_EPOCH, $BEST_R2"
        else
            echo "⏸  训练已停止: epoch $BEST_EPOCH, $BEST_R2 - 可 --resume"
        fi
    elif [ "$DATA_COUNT" -ge "$TARGET" ]; then
        echo "⏳ 训练未开始（数据已就绪）"
    else
        echo "⏸  训练未开始（等数据）"
    fi
    echo ""
}

resume() {
    DATA_COUNT=$(find ./data -name "*_params.json" 2>/dev/null | wc -l)
    LATEST_CKPT=$(ls -t checkpoints/best_model_*.pt 2>/dev/null | head -1)
    GEN_PID=$(pgrep -f "generate_dataset.py" | head -1)
    TRAIN_PID=$(pgrep -f "train.py" | head -1)

    # ============= 数据阶段 =============
    if [ "$DATA_COUNT" -lt "$TARGET" ]; then
        if [ -z "$GEN_PID" ]; then
            echo "⏯  继续数据生成（自动跳过 $DATA_COUNT 已有的）..."
            nohup python generate_dataset.py \
                --src-dir ./photos \
                --out-dir ./data \
                --variants-per-photo 30 \
                --img-size 384 \
                --n-workers 8 \
                > /tmp/data_gen.log 2>&1 &
            GEN_PID=$!
            echo "  数据生成已启动: PID $GEN_PID"
        else
            echo "✓ 数据生成正在运行 (PID $GEN_PID, $DATA_COUNT/$TARGET)"
        fi

        echo "⏳ 等待数据生成完成（每 60 秒检查一次）..."
        while true; do
            sleep 60
            COUNT=$(find ./data -name "*_params.json" 2>/dev/null | wc -l)
            PID_CHECK=$(pgrep -f "generate_dataset.py" | head -1)
            if [ "$COUNT" -ge "$TARGET" ]; then
                echo "  ✓ 数据完成: $COUNT 对"
                break
            fi
            if [ -z "$PID_CHECK" ]; then
                echo "  ⚠ 生成进程意外退出 ($COUNT/$TARGET)，重启..."
                nohup python generate_dataset.py \
                    --src-dir ./photos --out-dir ./data \
                    --variants-per-photo 30 --img-size 384 --n-workers 8 \
                    > /tmp/data_gen.log 2>&1 &
            else
                PCT=$((COUNT * 100 / TARGET))
                echo "  [$PCT%] $COUNT/$TARGET ($(date '+%H:%M:%S'))"
            fi
        done
        DATA_COUNT=$TARGET
        LATEST_CKPT=$(ls -t checkpoints/best_model_*.pt 2>/dev/null | head -1)
    fi

    # ============= 训练阶段 =============
    if [ -n "$TRAIN_PID" ]; then
        echo "✓ 训练正在运行 (PID $TRAIN_PID)，跳过"
        echo "  监控: tail -f /tmp/auto_train.log"
        return
    fi

    if [ -f checkpoints/test_results.json ]; then
        echo "✅ 训练已完成！查看结果:"
        cat checkpoints/test_results.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
m = d['test_metrics']
print(f'  R²:   {m[\"r2_mean\"]:.4f}')
print(f'  MAE:  {m[\"mae\"]:.4f}')"
        echo ""
        echo "💡 导出 ONNX 部署到 backend："
        echo "   python export_onnx.py"
        return
    fi

    # 启动训练（如有 checkpoint 则恢复）
    RESUME_ARG=""
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_ARG="--resume $LATEST_CKPT"
        echo "⏯  从 checkpoint 恢复训练: $LATEST_CKPT"
    else
        echo "🚀 启动新训练..."
    fi

    nohup python train.py \
        --data-dir ./data \
        --epochs 100 \
        --batch-size 16 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --backbone simple_color \
        --output-dir ./checkpoints \
        --device cuda \
        --num-workers 4 \
        --seed 42 \
        --param-loss-weight 1.0 \
        --pixel-loss-weight 1.0 \
        $RESUME_ARG \
        > /tmp/auto_train.log 2>&1 &
    echo "  训练已启动: PID $!"
    echo "  监控: tail -f /tmp/auto_train.log"
}

# Main
if [ "$1" = "--status" ]; then
    show_status
else
    show_status
    echo ""
    resume
fi
