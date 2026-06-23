#!/bin/bash
# 等数据完成 → 自动启动训练（v5: simple_color CNN + 双 loss）

set -e
cd "$(dirname "$0")"

TARGET=26280

echo "═══════════════════════════════════════════════════════"
echo "等待数据生成完成（目标 $TARGET 对）..."
echo "═══════════════════════════════════════════════════════"

while true; do
    count=$(find ./data -name "*_params.json" 2>/dev/null | wc -l)
    gen_pid=$(ps aux | grep "generate_dataset.py" | grep -v grep | awk '{print $2}' | head -1)

    if [ "$count" -ge "$TARGET" ] || [ -z "$gen_pid" ]; then
        echo "✓ 数据准备完成: $count 对"
        break
    fi

    pct=$((count * 100 / TARGET))
    echo "  [$pct%] $count/$TARGET ($(date '+%H:%M:%S'))"
    sleep 60
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "🚀 启动训练（simple_color CNN + 双 loss）"
echo "═══════════════════════════════════════════════════════"

rm -rf checkpoints
python train.py \
    --data-dir ./data \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --backbone simple_color \
    --output-dir ./checkpoints \
    --device cuda \
    --num-workers 4 \
    --seed 42 \
    --param-loss-weight 1.0 \
    --pixel-loss-weight 1.0

echo ""
echo "═══════════════════════════════════════════════════════"
echo "✅ 训练完成！查看结果："
echo "═══════════════════════════════════════════════════════"
cat ./checkpoints/test_results.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
m = data['test_metrics']
print(f'MAE: {m[\"mae\"]:.4f}')
print(f'R²:  {m[\"r2_mean\"]:.4f}')
"
