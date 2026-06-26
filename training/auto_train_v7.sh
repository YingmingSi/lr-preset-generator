#!/bin/bash
# 等数据完成 → 自动启动 5-stage 课程训练（v7: 72 维参数）
set -e
cd "$(dirname "$0")"

TARGET=43800

echo "═══════════════════════════════════════════════════════"
echo "等待数据生成完成（目标 $TARGET 对）..."
echo "═══════════════════════════════════════════════════════"

while true; do
    count=$(find ./data -name "*_params.json" 2>/dev/null | wc -l)
    gen_pid=$(pgrep -f "generate_dataset.py" | head -1)
    if [ "$count" -ge "$TARGET" ] || [ -z "$gen_pid" ]; then
        echo "✓ 数据准备完成: $count 对"
        break
    fi
    echo "  [$((count * 100 / TARGET))%] $count/$TARGET ($(date '+%H:%M:%S'))"
    sleep 60
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🚀 启动 5-stage 课程训练（72 维参数）"
echo "═══════════════════════════════════════════════════════"

rm -rf checkpoints
python train.py \
    --data-dir ./data \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --backbone simple_color \
    --output-dir ./checkpoints \
    --device cuda \
    --num-workers 4 \
    --seed 42 \
    --stage-epochs 50,50,50,30,30 \
    --pixel-loss-weight 0.3

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ 训练完成！结果："
echo "═══════════════════════════════════════════════════════"
cat ./checkpoints/test_results.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
m = d['test_metrics']
print(f'  R² (全72): {m[\"r2_mean\"]:.4f}')
print(f'  MAE:       {m[\"mae\"]:.4f}')
"
