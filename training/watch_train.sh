#!/bin/bash
# 训练监控脚本
# 用法:
#   ./watch_train.sh            # 单次显示当前状态
#   ./watch_train.sh --live     # 实时跟踪所有训练输出
#   ./watch_train.sh --summary  # 显示每个 epoch 的关键指标
#   ./watch_train.sh --stop     # 停止训练

LOG=/tmp/train.log
PID_PATTERN="train.py.*convnext_tiny"

case "${1:-status}" in
    status|"")
        echo "═══════════════════════════════════════════════════════"
        echo "📊 训练状态  $(date '+%H:%M:%S')"
        echo "═══════════════════════════════════════════════════════"

        # 检查进程
        if pgrep -f "$PID_PATTERN" > /dev/null; then
            PID=$(pgrep -f "$PID_PATTERN" | head -1)
            UPTIME=$(ps -o etime= -p $PID 2>/dev/null | tr -d ' ')
            echo "✓ 训练运行中 (PID: $PID, 已运行: $UPTIME)"
        else
            echo "❌ 训练未运行"
        fi

        # 最近 epoch
        LATEST=$(grep "验证损失" $LOG 2>/dev/null | tail -1)
        if [ -n "$LATEST" ]; then
            echo ""
            echo "📈 最新 epoch:"
            echo "  $LATEST"
        fi

        # 已完成 epoch 数
        EPOCHS=$(grep -c "训练损失" $LOG 2>/dev/null)
        echo ""
        echo "  已完成: $EPOCHS epochs"

        # GPU 状态
        GPU_INFO=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -1)
        echo "  GPU: $GPU_INFO"

        # 最佳验证 R²
        BEST=$(grep "保存到" $LOG 2>/dev/null | tail -1 | grep -oP "epoch\d+_r2[\d.\-]+" | tail -1)
        if [ -n "$BEST" ]; then
            echo "  最佳: $BEST"
        fi

        echo ""
        echo "命令选项:"
        echo "  ./watch_train.sh --live      # 实时跟踪所有输出（Ctrl+C 退出）"
        echo "  ./watch_train.sh --summary   # 显示每个 epoch 摘要"
        echo "  ./watch_train.sh --stop      # 停止训练"
        ;;

    --live)
        echo "实时跟踪训练日志（按 Ctrl+C 退出）"
        echo "═══════════════════════════════════════════════════════"
        tail -f $LOG
        ;;

    --summary)
        echo "═══════════════════════════════════════════════════════"
        echo "📊 训练历史摘要（每 epoch 一行）"
        echo "═══════════════════════════════════════════════════════"
        echo ""
        printf "%-8s %-15s %-15s %-12s\n" "Epoch" "训练损失" "验证损失" "R²"
        echo "─────────────────────────────────────────────────"

        awk '
        /Epoch [0-9]+\/[0-9]+/ {
            match($0, /Epoch ([0-9]+)\/[0-9]+/, arr)
            epoch = arr[1]
        }
        /训练损失/ {
            match($0, /训练损失: ([0-9.]+)/, arr)
            train_loss = arr[1]
        }
        /验证损失/ {
            match($0, /验证损失: ([0-9.]+).*R²: ([\-0-9.]+)/, arr)
            val_loss = arr[1]
            r2 = arr[2]
            printf "%-8s %-15s %-15s %-12s\n", epoch, train_loss, val_loss, r2
        }' $LOG
        ;;

    --stop)
        if pgrep -f "$PID_PATTERN" > /dev/null; then
            echo "⚠️  确定要停止训练？(y/N)"
            read -r CONFIRM
            if [ "$CONFIRM" = "y" ] || [ "$CONFIRM" = "Y" ]; then
                pkill -f "$PID_PATTERN"
                echo "✓ 训练已停止"
            else
                echo "已取消"
            fi
        else
            echo "训练未运行"
        fi
        ;;

    --help|-h)
        echo "用法: $0 [选项]"
        echo ""
        echo "选项:"
        echo "  (无参数)        显示当前状态（PID、最新 epoch、GPU）"
        echo "  --live          实时跟踪日志（tail -f）"
        echo "  --summary       表格显示所有 epoch 摘要"
        echo "  --stop          停止训练"
        echo "  --help          显示此帮助"
        ;;

    *)
        echo "未知选项: $1"
        echo "运行 $0 --help 查看用法"
        exit 1
        ;;
esac
