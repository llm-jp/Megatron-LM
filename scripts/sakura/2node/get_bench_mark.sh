#!/bin/bash

# パスと変数の設定
WORK_DIR="/mnt/taishi-work-space/Megatron-LM"
SCRIPT_DIR="${WORK_DIR}/scripts/sakura/2node"
# SCRIPT_NAME="llama-2-13b-base.sh"
# SCRIPT_NAME="gpt-7b-base-selective.sh"
# SCRIPT_NAME="gpt-13b-base-selective-transformer-engine.sh"
SCRIPT_NAME="gpt-13b-base-selective.sh"
LOG_DIR="${WORK_DIR}/bench_mark_log"

# scripts/sakura/2node/gpt-13b-base-selective.sh

# 引数の設定
SEQ_LENGTH=$1            # 第1引数: シーケンス長
TENSOR_PARALLEL_SIZE=$2  # 第2引数: テンソル並列サイズ
PIPELINE_PARALLEL_SIZE=$3  # 第3引数: パイプライン並列サイズ
MICRO_BATCH_SIZE=$4      # 第4引数: マイクロバッチサイズ

# ログファイルパスの生成
# LOG_FILE="${LOG_DIR}/transformer/${SCRIPT_NAME}_SEQ_LENGTH_${SEQ_LENGTH}_TENSOR_PARALLEL_SIZE_${TENSOR_PARALLEL_SIZE}_PIPELINE_PARALLEL_SIZE_${PIPELINE_PARALLEL_SIZE}_MICRO_BATCH_SIZE_${MICRO_BATCH_SIZE}"
CURRENT_DATE=$(date +%Y-%m-%d_%H-%M-%S)
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_SEQ_LENGTH_${SEQ_LENGTH}_TENSOR_PARALLEL_SIZE_${TENSOR_PARALLEL_SIZE}_PIPELINE_PARALLEL_SIZE_${PIPELINE_PARALLEL_SIZE}_MICRO_BATCH_SIZE_${MICRO_BATCH_SIZE}_${CURRENT_DATE}"

echo "LOG_FILE $LOG_FILE"

# スクリプトパスの生成
SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_NAME}"

# ログディレクトリを確認し、存在しない場合は作成
mkdir -p "${LOG_DIR}"

# コマンドの実行
bash "${SCRIPT_PATH}" "${SEQ_LENGTH}" "${TENSOR_PARALLEL_SIZE}" "${PIPELINE_PARALLEL_SIZE}" "${MICRO_BATCH_SIZE}" >> "${LOG_FILE}" 2>&1
