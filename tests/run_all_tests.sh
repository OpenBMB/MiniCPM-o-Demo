#!/bin/bash
# MiniCPMO45 Service 完整测试脚本
#
# 运行所有测试并生成 case 输出到 tests/resources/output/
#
# 用法:
#   cd /user/sunweiyue/lib/swy-dev/minicpmo45_service
#   bash tests/run_all_tests.sh [options]
#
# 选项:
#   --clean    清理旧的输出目录后重新生成
#   --quick    只运行 schema 测试（无需 GPU）
#   --gpu N    指定 GPU 编号（默认 0）

# 不使用 set -e，允许测试失败后继续运行其他测试
FAILED_TESTS=()

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# 默认参数
CLEAN=false
QUICK=false
GPU=0

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "MiniCPMO45 Service 测试"
echo "========================================"
echo "项目目录: $PROJECT_ROOT"
echo "GPU: $GPU"
echo ""

# 清理输出目录
if [ "$CLEAN" = true ]; then
    echo "[1/4] 清理旧的输出目录..."
    rm -rf tests/results/chat/case_*
    rm -rf tests/results/streaming/case_*
    rm -rf tests/results/duplex/case_*
    echo "✓ 清理完成"
else
    echo "[1/4] 跳过清理（使用 --clean 强制清理）"
fi

# 运行 schema 测试（无需 GPU）
echo ""
echo "[2/5] 运行 Schema 测试..."
if PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_schemas.py -v --tb=short; then
    echo "✓ Schema 测试通过"
else
    echo "✗ Schema 测试失败"
    FAILED_TESTS+=("schema")
fi

if [ "$QUICK" = true ]; then
    echo ""
    echo "========================================"
    echo "快速模式：跳过 GPU 测试"
    echo "========================================"
    if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
        exit 1
    fi
    exit 0
fi

# 运行 Chat 测试（需要 GPU）
echo ""
echo "[3/5] 运行 Chat 测试（需要 GPU）..."
if CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_chat.py -v -s --tb=short; then
    echo "✓ Chat 测试通过"
else
    echo "✗ Chat 测试失败"
    FAILED_TESTS+=("chat")
fi

# 运行 Streaming 测试（需要 GPU）
echo ""
echo "[4/5] 运行 Streaming 测试（需要 GPU）..."
if CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_streaming.py -v -s --tb=short; then
    echo "✓ Streaming 测试通过"
else
    echo "✗ Streaming 测试失败"
    FAILED_TESTS+=("streaming")
fi

# 运行 Duplex 测试（需要 GPU）
echo ""
echo "[5/5] 运行 Duplex 测试（需要 GPU）..."
if CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. .venv/base/bin/python -m pytest tests/test_duplex.py -v -s --tb=short; then
    echo "✓ Duplex 测试通过"
else
    echo "✗ Duplex 测试失败"
    FAILED_TESTS+=("duplex")
fi

# 统计结果
echo ""
echo "========================================"
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "所有测试通过！"
else
    echo "测试完成（有失败）"
fi
echo "========================================"
echo ""

# 显示失败的测试
if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "失败的测试模块: ${FAILED_TESTS[*]}"
    echo ""
fi

echo "生成的 case 目录（按 processor 类型组织）:"
for processor in chat streaming duplex; do
    if [ -d "tests/results/$processor" ]; then
        echo "  $processor/:"
        ls -1 "tests/results/$processor" 2>/dev/null | grep "^case_" | while read dir; do
            echo "    - $dir"
        done
    fi
done
echo ""
echo "查看 case 内容:"
echo "  cat tests/results/<processor>/case_<name>/input.json"
echo "  cat tests/results/<processor>/case_<name>/output.json"

# 返回正确的 exit code
if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    exit 1
fi
exit 0
