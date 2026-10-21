#!/bin/bash
#SBATCH --job-name=regraph_ray_fused
#SBATCH --partition=workq
#SBATCH --exclusive
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=384
#SBATCH --mem=0
#SBATCH --time=23:00:00
#SBATCH --output=/scratch/zhoul0e/ReGraphv2/Scripts/ray_opt_ablation/slurm_logs/regraph_ray_fused-%j.out
#SBATCH --error=/scratch/zhoul0e/ReGraphv2/Scripts/ray_opt_ablation/slurm_logs/regraph_ray_fused-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scratch/zhoul0e/ReGraphv2}"
WORKDIR="${REPO_ROOT}"
IMAGE="${IMAGE:-/scratch/zhoul0e/regraph-data-env-llvm18.1.3.sif}"
BIND_ROOT="${BIND_ROOT:-/scratch/zhoul0e:/scratch/zhoul0e}"
DRIVER="${REPO_ROOT}/Scripts/ray_opt_ablation/ray_fused_pipeline.py"
OPT_LEVEL="${1:-${OPT_LEVEL:-}}"
SMOKE="${SMOKE:-0}"
DATASET_PATH="${DATASET_PATH:-}"
DEFAULT_DATASET_PATH="${DEFAULT_DATASET_PATH:-/scratch/zhoul0e/Dataset-1}"
DEFAULT_SMOKESET_PATH="${DEFAULT_SMOKESET_PATH:-/scratch/zhoul0e/Dataset-smoketest}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
RESUME="${RESUME:-0}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"
FINAL_FILTER_REFERENCE="${FINAL_FILTER_REFERENCE:-${REGRAPH_FINAL_FILTER_REFERENCE:-${FINAL_CSV_FILTER_DIR:-${REGRAPH_FINAL_CSV_FILTER_DIR:-}}}}"
FINAL_FILTER_REFERENCE_KIND="${FINAL_FILTER_REFERENCE_KIND:-${REGRAPH_FINAL_FILTER_REFERENCE_KIND:-auto}}"
FINAL_FILTER_MATCH_MODE="${FINAL_FILTER_MATCH_MODE:-${REGRAPH_FINAL_FILTER_MATCH_MODE:-exact}}"
FINAL_UNINFORMATIVE_FILTER="${FINAL_UNINFORMATIVE_FILTER:-${REGRAPH_FINAL_UNINFORMATIVE_FILTER:-1}}"
FINAL_UNINFORMATIVE_FILTER_SPLITS="${FINAL_UNINFORMATIVE_FILTER_SPLITS:-${REGRAPH_FINAL_UNINFORMATIVE_FILTER_SPLITS:-train,validation,test}}"
FINAL_UNINFORMATIVE_MAX_STUB_TOKENS="${FINAL_UNINFORMATIVE_MAX_STUB_TOKENS:-${REGRAPH_FINAL_UNINFORMATIVE_MAX_STUB_TOKENS:-128}}"
FINAL_UNINFORMATIVE_COLLISION_MIN_ROWS="${FINAL_UNINFORMATIVE_COLLISION_MIN_ROWS:-${REGRAPH_FINAL_UNINFORMATIVE_COLLISION_MIN_ROWS:-50}}"
FINAL_UNINFORMATIVE_COLLISION_MIN_GROUPS="${FINAL_UNINFORMATIVE_COLLISION_MIN_GROUPS:-${REGRAPH_FINAL_UNINFORMATIVE_COLLISION_MIN_GROUPS:-20}}"
TASK3_RECORD_FORMAT="${TASK3_RECORD_FORMAT:-${REGRAPH_TASK3_RECORD_FORMAT:-graph}}"
SPLITS="${SPLITS:-${REGRAPH_SPLITS:-train,validation,test}}"
DRIVER_EXTRA_ARGS="${DRIVER_EXTRA_ARGS:-}"
WORKER_PORT_SPAN="${WORKER_PORT_SPAN:-900}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/regraph_${SLURM_JOB_ID}}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:-384}"
nodes_array=()
ray_step_pids=()
ray_step_labels=()
LOG_ROOT="${LOG_ROOT:-/scratch/zhoul0e/regraphv2_logs/slurm_ray_fused_pipeline}"
ray_step_state_dir="${LOG_ROOT}/ray_step_state_${SLURM_JOB_ID}"
ray_step_failure_dir="${ray_step_state_dir}/failures"
ray_cleanup_marker="${ray_step_state_dir}/cleanup_started"

if [ -z "${DATASET_PATH}" ]; then
    if [ "${SMOKE}" = "1" ]; then
        DATASET_PATH="${DEFAULT_SMOKESET_PATH}"
    else
        DATASET_PATH="${DEFAULT_DATASET_PATH}"
    fi
fi

cd "${WORKDIR}"
mkdir -p "${LOG_ROOT}"
rm -rf "${ray_step_state_dir}"
mkdir -p "${ray_step_failure_dir}"

export REGRAPH_CACHE_ROOT="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_ASSETS_CACHE="${CACHE_ROOT}/huggingface/assets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export RAY_TMPDIR="/tmp/regraph_${SLURM_JOB_ID}"
export TMPDIR="${RAY_TMPDIR}"
mkdir -p \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${HF_HUB_CACHE}" \
    "${HF_ASSETS_CACHE}" \
    "${TRANSFORMERS_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${TMPDIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_on_node() {
    local node="$1"
    shift
    srun --overlap --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" -w "${node}" "$@"
}

run_in_container_on_node() {
    local node="$1"
    shift
    run_on_node "${node}" singularity exec --bind "${BIND_ROOT}" "${IMAGE}" "$@"
}

start_background_ray_step() {
    local node="$1"
    local label="$2"
    local command="$3"
    local safe_label="${label//[^A-Za-z0-9_.@-]/_}"

    run_in_container_on_node "${node}" /bin/bash -c "
set +e
${command}
status=\$?
if [ -f '${ray_cleanup_marker}' ]; then
    echo \"[ray-step] ${label} exited after cleanup with status=\${status}\"
elif [ \"\${status}\" -eq 0 ]; then
    echo \"[ray-step] ${label} exited cleanly\"
else
    echo \"[ray-step] ${label} exited with status=\${status}\"
fi
if [ ! -f '${ray_cleanup_marker}' ]; then
    echo \"status=\${status}\" > '${ray_step_failure_dir}/${safe_label}.failed'
fi
exit 0
" &
    ray_step_pids+=("$!")
    ray_step_labels+=("${label}")
}

check_background_ray_steps() {
    local failed=0

    if compgen -G "${ray_step_failure_dir}/*.failed" >/dev/null; then
        log "ERROR: at least one Ray background step exited early"
        for failure_file in "${ray_step_failure_dir}"/*.failed; do
            log "ERROR: $(basename "${failure_file}") $(cat "${failure_file}")"
        done
        failed=1
    fi

    for index in "${!ray_step_pids[@]}"; do
        local pid="${ray_step_pids[${index}]}"
        local label="${ray_step_labels[${index}]}"
        if ! kill -0 "${pid}" >/dev/null 2>&1; then
            log "ERROR: Ray background step exited before driver completion: ${label} pid=${pid}"
            failed=1
        fi
    done

    return "${failed}"
}

cleanup_ray() {
    local restore_errexit=0
    local still_running_pids=()

    case "$-" in
        *e*) restore_errexit=1 ;;
    esac
    set +e

    if [ "${#ray_step_pids[@]}" -gt 0 ]; then
        touch "${ray_cleanup_marker}" 2>/dev/null || true
    fi

    log "cleanup: stopping Ray on all nodes"
    for node in "${nodes_array[@]}"; do
        log "cleanup: ray stop --force on ${node}"
        run_in_container_on_node "${node}" ray stop --force >/dev/null 2>&1 || true
    done

    if [ "${#ray_step_pids[@]}" -gt 0 ]; then
        sleep 3
        for pid in "${ray_step_pids[@]}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                still_running_pids+=("${pid}")
            fi
        done

        if [ "${#still_running_pids[@]}" -gt 0 ]; then
            log "cleanup: terminating background srun steps: ${still_running_pids[*]}"
            kill "${still_running_pids[@]}" >/dev/null 2>&1 || true
        fi

        wait "${ray_step_pids[@]}" >/dev/null 2>&1 || true
    fi

    if [ "${restore_errexit}" -eq 1 ]; then
        set -e
    fi
}

trap cleanup_ray EXIT

if [ -z "${OPT_LEVEL}" ]; then
    log "ERROR: missing OPT_LEVEL. Usage: sbatch $0 <OPT_LEVEL>"
    exit 1
fi

log "loading Singularity module"
module load singularity >/dev/null 2>&1

if ! command -v singularity >/dev/null 2>&1; then
    log "ERROR: singularity is not available after module load"
    exit 1
fi

if [ ! -f "${IMAGE}" ]; then
    log "ERROR: image not found: ${IMAGE}"
    exit 1
fi

if [ ! -f "${DRIVER}" ]; then
    log "ERROR: driver not found: ${DRIVER}"
    exit 1
fi

required_pass_libs=(
    "${REPO_ROOT}/GraphBuilder/meta_remover/build/libStripAllMetadataPass.so"
    "${REPO_ROOT}/GraphBuilder/cfg_exporter/build/libMyCFGPrinterPass.so"
    "${REPO_ROOT}/GraphBuilder/ddg_exporter/build/libDDGPrinter.so"
)
missing_pass_lib=0
for pass_lib in "${required_pass_libs[@]}"; do
    if [ ! -f "${pass_lib}" ]; then
        missing_pass_lib=1
        log "missing GraphBuilder pass library: ${pass_lib}"
    fi
done
if [ "${missing_pass_lib}" -eq 1 ]; then
    log "building GraphBuilder LLVM pass libraries inside container"
    singularity exec --bind "${BIND_ROOT}" "${IMAGE}" /bin/bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
for d in GraphBuilder/meta_remover GraphBuilder/cfg_exporter GraphBuilder/ddg_exporter; do
    echo \"[build-pass] \${d}\"
    cmake -S \"\${d}\" -B \"\${d}/build\" -G Ninja -DLLVM_DIR=\$(llvm-config --cmakedir)
    cmake --build \"\${d}/build\" -j 8
done
"
fi

nodes=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
readarray -t nodes_array <<<"${nodes}"
EXPECTED_NODES="${#nodes_array[@]}"
head_node="${nodes_array[0]}"
head_ip=$(run_on_node "${head_node}" hostname -I | awk '{print $1}')

base_port=$((20000 + SLURM_JOB_ID % 20000))
ray_port="${base_port}"
ray_dashboard_port=$((base_port + 1))
head_node_manager_port=$((base_port + 2))
head_object_manager_port=$((base_port + 3))
head_runtime_env_agent_port=$((base_port + 4))
head_min_worker_port=$((base_port + 100))
head_max_worker_port=$((head_min_worker_port + WORKER_PORT_SPAN - 1))
ray_address="${head_ip}:${ray_port}"
expected_cpus=$((EXPECTED_NODES * CPUS_PER_NODE))

log "job_id=${SLURM_JOB_ID}"
log "repo_root=${REPO_ROOT}"
log "node_list=${SLURM_JOB_NODELIST}"
log "nodes=${nodes_array[*]}"
log "head_node=${head_node}"
log "head_ip=${head_ip}"
log "ray_address=${ray_address}"
log "dashboard_port=${ray_dashboard_port}"
log "expected_nodes=${EXPECTED_NODES}"
log "cpus_per_node=${CPUS_PER_NODE}"
log "expected_total_cpus=${expected_cpus}"
log "worker_port_span=${WORKER_PORT_SPAN}"
log "head_worker_port_range=${head_min_worker_port}-${head_max_worker_port}"
log "opt_level=${OPT_LEVEL}"
log "dataset_path=${DATASET_PATH}"
log "output_path=${OUTPUT_PATH}"
log "cache_root=${CACHE_ROOT}"
log "hf_datasets_cache=${HF_DATASETS_CACHE}"
log "final_filter_reference=${FINAL_FILTER_REFERENCE:-disabled}"
log "final_filter_reference_kind=${FINAL_FILTER_REFERENCE_KIND}"
log "final_filter_match_mode=${FINAL_FILTER_MATCH_MODE}"
log "final_uninformative_filter=${FINAL_UNINFORMATIVE_FILTER}"
log "final_uninformative_filter_splits=${FINAL_UNINFORMATIVE_FILTER_SPLITS}"
log "final_uninformative_max_stub_tokens=${FINAL_UNINFORMATIVE_MAX_STUB_TOKENS}"
log "final_uninformative_collision_min_rows=${FINAL_UNINFORMATIVE_COLLISION_MIN_ROWS}"
log "final_uninformative_collision_min_groups=${FINAL_UNINFORMATIVE_COLLISION_MIN_GROUPS}"
log "task3_record_format=${TASK3_RECORD_FORMAT}"
log "splits=${SPLITS}"
log "smoke=${SMOKE} resume=${RESUME} force_clean=${FORCE_CLEAN}"
log "driver_extra_args=${DRIVER_EXTRA_ARGS}"

log "pre-cleaning any stale Ray processes on allocated nodes"
cleanup_ray

log "starting Ray head on ${head_node}"
start_background_ray_step "${head_node}" "head@${head_node}" \
    "ray start \
        --head \
        --temp-dir='${RAY_TMPDIR}' \
        --node-ip-address='${head_ip}' \
        --port='${ray_port}' \
        --dashboard-host=0.0.0.0 \
        --dashboard-port='${ray_dashboard_port}' \
        --node-manager-port='${head_node_manager_port}' \
        --object-manager-port='${head_object_manager_port}' \
        --runtime-env-agent-port='${head_runtime_env_agent_port}' \
        --min-worker-port='${head_min_worker_port}' \
        --max-worker-port='${head_max_worker_port}' \
        --num-cpus='${CPUS_PER_NODE}' \
        --disable-usage-stats \
        --block"

sleep 10

for worker_index in "${!nodes_array[@]}"; do
    if [ "${worker_index}" -eq 0 ]; then
        continue
    fi

    worker_node="${nodes_array[${worker_index}]}"
    worker_ip=$(run_on_node "${worker_node}" hostname -I | awk '{print $1}')
    worker_base_port=$((base_port + 1000 + (worker_index - 1) * 1000))
    worker_node_manager_port=$((worker_base_port + 2))
    worker_object_manager_port=$((worker_base_port + 3))
    worker_runtime_env_agent_port=$((worker_base_port + 4))
    worker_min_worker_port=$((worker_base_port + 100))
    worker_max_worker_port=$((worker_min_worker_port + WORKER_PORT_SPAN - 1))

    log "starting Ray worker on ${worker_node} with ip=${worker_ip} worker_port_range=${worker_min_worker_port}-${worker_max_worker_port}"
    start_background_ray_step "${worker_node}" "worker@${worker_node}" \
        "ray start \
            --address='${ray_address}' \
            --temp-dir='${RAY_TMPDIR}' \
            --node-ip-address='${worker_ip}' \
            --node-manager-port='${worker_node_manager_port}' \
            --object-manager-port='${worker_object_manager_port}' \
            --runtime-env-agent-port='${worker_runtime_env_agent_port}' \
            --min-worker-port='${worker_min_worker_port}' \
            --max-worker-port='${worker_max_worker_port}' \
            --num-cpus='${CPUS_PER_NODE}' \
            --disable-usage-stats \
            --block"
done

sleep 15

log "ray status from head node"
run_in_container_on_node "${head_node}" ray status --address="${ray_address}"
if ! check_background_ray_steps; then
    log "ERROR: at least one Ray node exited before driver submission"
    exit 1
fi

driver_args=(
    --repo-root "${REPO_ROOT}"
    --dataset-path "${DATASET_PATH}"
    --cache-root "${CACHE_ROOT}"
    --opt-level "${OPT_LEVEL}"
    --task3-record-format "${TASK3_RECORD_FORMAT}"
    --splits "${SPLITS}"
)

if [ -n "${OUTPUT_PATH}" ]; then
    driver_args+=(--output-path "${OUTPUT_PATH}")
fi
if [ "${SMOKE}" = "1" ]; then
    driver_args+=(--smoke)
fi
if [ "${RESUME}" = "1" ]; then
    driver_args+=(--resume)
fi
if [ "${FORCE_CLEAN}" = "1" ]; then
    driver_args+=(--force-clean)
fi
if [ -n "${FINAL_FILTER_REFERENCE}" ]; then
    driver_args+=(
        --final-filter-reference "${FINAL_FILTER_REFERENCE}"
        --final-filter-reference-kind "${FINAL_FILTER_REFERENCE_KIND}"
        --final-filter-match-mode "${FINAL_FILTER_MATCH_MODE}"
    )
fi
if [ "${FINAL_UNINFORMATIVE_FILTER}" = "1" ]; then
    driver_args+=(
        --final-uninformative-filter
        --final-uninformative-filter-splits "${FINAL_UNINFORMATIVE_FILTER_SPLITS}"
        --final-uninformative-max-stub-tokens "${FINAL_UNINFORMATIVE_MAX_STUB_TOKENS}"
        --final-uninformative-collision-min-rows "${FINAL_UNINFORMATIVE_COLLISION_MIN_ROWS}"
        --final-uninformative-collision-min-groups "${FINAL_UNINFORMATIVE_COLLISION_MIN_GROUPS}"
    )
fi
if [ -n "${DRIVER_EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    extra_args=(${DRIVER_EXTRA_ARGS})
    driver_args+=("${extra_args[@]}")
fi

log "running Ray fused pipeline driver on head node (bypassing ray job submit API)"
set +e
run_in_container_on_node "${head_node}" \
    env \
        RAY_ADDRESS="${ray_address}" \
        REGRAPH_REPO_ROOT="${REPO_ROOT}" \
        REGRAPH_CACHE_ROOT="${CACHE_ROOT}" \
        HF_HOME="${HF_HOME}" \
        HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
        HF_HUB_CACHE="${HF_HUB_CACHE}" \
        HF_ASSETS_CACHE="${HF_ASSETS_CACHE}" \
        TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
        XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
        RAY_TMPDIR="${RAY_TMPDIR}" \
        TMPDIR="${TMPDIR}" \
        REGRAPH_TOKENIZER_PATH="${REPO_ROOT}/Tokenizer/output_tokenizer/llvm_ir_bpe.json" \
        REGRAPH_FINAL_FILTER_REFERENCE="${FINAL_FILTER_REFERENCE}" \
        REGRAPH_FINAL_FILTER_REFERENCE_KIND="${FINAL_FILTER_REFERENCE_KIND}" \
        REGRAPH_FINAL_FILTER_MATCH_MODE="${FINAL_FILTER_MATCH_MODE}" \
        REGRAPH_FINAL_UNINFORMATIVE_FILTER="${FINAL_UNINFORMATIVE_FILTER}" \
        REGRAPH_FINAL_UNINFORMATIVE_FILTER_SPLITS="${FINAL_UNINFORMATIVE_FILTER_SPLITS}" \
        REGRAPH_FINAL_UNINFORMATIVE_MAX_STUB_TOKENS="${FINAL_UNINFORMATIVE_MAX_STUB_TOKENS}" \
        REGRAPH_FINAL_UNINFORMATIVE_COLLISION_MIN_ROWS="${FINAL_UNINFORMATIVE_COLLISION_MIN_ROWS}" \
        REGRAPH_FINAL_UNINFORMATIVE_COLLISION_MIN_GROUPS="${FINAL_UNINFORMATIVE_COLLISION_MIN_GROUPS}" \
        REGRAPH_TASK3_RECORD_FORMAT="${TASK3_RECORD_FORMAT}" \
        REGRAPH_SPLITS="${SPLITS}" \
        python3 "${DRIVER}" "${driver_args[@]}"
driver_status=$?
set -e

if [ "${driver_status}" -ne 0 ]; then
    log "ERROR: driver execution failed status=${driver_status}"
    exit 1
fi

log "job finished successfully"
