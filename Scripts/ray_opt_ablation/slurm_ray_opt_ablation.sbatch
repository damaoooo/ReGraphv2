#!/bin/bash
#SBATCH --job-name=regraph_ray_opt
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-23:00:00
#SBATCH --output=/ibex/tmp/zhoul0e/regraphv2/Scripts/ray_opt_ablation/slurm_logs/slurm-%j.out
#SBATCH --error=/ibex/tmp/zhoul0e/regraphv2/Scripts/ray_opt_ablation/slurm_logs/slurm-%j.err

set -euo pipefail

REPO_ROOT="/ibex/tmp/zhoul0e/regraphv2"
WORKDIR="${REPO_ROOT}"
IMAGE="/ibex/tmp/zhoul0e/regraph-data-env-llvm18.1.3.sif"
BIND_ROOT="/ibex/tmp/zhoul0e:/ibex/tmp/zhoul0e"
DRIVER="${REPO_ROOT}/Scripts/ray_opt_ablation/ray_opt_ablation.py"
OPT_LEVEL="${1:-${OPT_LEVEL:-}}"
DATASET_PATH="${DATASET_PATH:-/ibex/tmp/zhoul0e/Dataset-1}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
SMOKE="${SMOKE:-0}"
RESUME="${RESUME:-0}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"
DRIVER_EXTRA_ARGS="${DRIVER_EXTRA_ARGS:-}"
CACHE_ROOT="${CACHE_ROOT:-/ibex/tmp/zhoul0e/regraph_cache/${SLURM_JOB_ID}}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK}"
nodes_array=()
ray_step_pids=()
ray_step_labels=()
ray_step_state_dir="${REPO_ROOT}/Scripts/ray_opt_ablation/slurm_logs/ray_step_state_${SLURM_JOB_ID}"
ray_step_failure_dir="${ray_step_state_dir}/failures"
ray_cleanup_marker="${ray_step_state_dir}/cleanup_started"

cd "${WORKDIR}"
mkdir -p "${REPO_ROOT}/Scripts/ray_opt_ablation/slurm_logs"
rm -rf "${ray_step_state_dir}"
mkdir -p "${ray_step_failure_dir}"

export REGRAPH_CACHE_ROOT="${CACHE_ROOT}"
export HF_HOME="${CACHE_ROOT}/huggingface"
export HF_DATASETS_CACHE="${CACHE_ROOT}/huggingface/datasets"
export HF_HUB_CACHE="${CACHE_ROOT}/huggingface/hub"
export HF_ASSETS_CACHE="${CACHE_ROOT}/huggingface/assets"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/huggingface/transformers"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export TMPDIR="${CACHE_ROOT}/tmp"
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
if [ \"\${status}\" -eq 0 ]; then
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
    set +e
    local still_running_pids=()

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

nodes=$(scontrol show hostnames "${SLURM_JOB_NODELIST}")
readarray -t nodes_array <<<"${nodes}"
EXPECTED_NODES="${#nodes_array[@]}"
head_node="${nodes_array[0]}"
head_ip=$(run_on_node "${head_node}" hostname -I | awk '{print $1}')

base_port=$((20000 + SLURM_JOB_ID % 20000))
ray_port="${base_port}"
ray_dashboard_port=$((base_port + 1))
node_manager_port=$((base_port + 2))
object_manager_port=$((base_port + 3))
runtime_env_agent_port=$((base_port + 4))
min_worker_port=$((base_port + 100))
max_worker_port=$((base_port + 199))
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
log "opt_level=${OPT_LEVEL}"
log "dataset_path=${DATASET_PATH}"
log "output_path=${OUTPUT_PATH}"
log "cache_root=${CACHE_ROOT}"
log "hf_datasets_cache=${HF_DATASETS_CACHE}"
log "smoke=${SMOKE} resume=${RESUME} force_clean=${FORCE_CLEAN}"
log "driver_extra_args=${DRIVER_EXTRA_ARGS}"

log "pre-cleaning any stale Ray processes on allocated nodes"
cleanup_ray

log "starting Ray head on ${head_node}"
start_background_ray_step "${head_node}" "head@${head_node}" \
    "ray start \
        --head \
        --node-ip-address='${head_ip}' \
        --port='${ray_port}' \
        --dashboard-host=0.0.0.0 \
        --dashboard-port='${ray_dashboard_port}' \
        --node-manager-port='${node_manager_port}' \
        --object-manager-port='${object_manager_port}' \
        --runtime-env-agent-port='${runtime_env_agent_port}' \
        --min-worker-port='${min_worker_port}' \
        --max-worker-port='${max_worker_port}' \
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

    log "starting Ray worker on ${worker_node} with ip=${worker_ip}"
    start_background_ray_step "${worker_node}" "worker@${worker_node}" \
        "ray start \
            --address='${ray_address}' \
            --node-ip-address='${worker_ip}' \
            --node-manager-port='${node_manager_port}' \
            --object-manager-port='${object_manager_port}' \
            --runtime-env-agent-port='${runtime_env_agent_port}' \
            --min-worker-port='${min_worker_port}' \
            --max-worker-port='${max_worker_port}' \
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
if [ -n "${DRIVER_EXTRA_ARGS}" ]; then
    # shellcheck disable=SC2206
    extra_args=(${DRIVER_EXTRA_ARGS})
    driver_args+=("${extra_args[@]}")
fi

ray_job_id="regraph-ray-opt-${SLURM_JOB_ID}"
ray_job_address="http://${head_ip}:${ray_dashboard_port}"

log "submitting Ray opt ablation driver"
set +e
run_in_container_on_node "${head_node}" \
    ray job submit \
        --address="${ray_job_address}" \
        --submission-id="${ray_job_id}" \
        --entrypoint-num-cpus=1 \
        --entrypoint-resources="{\"node:${head_ip}\": 0.001}" \
        -- \
        env \
            RAY_ADDRESS="auto" \
            REGRAPH_REPO_ROOT="${REPO_ROOT}" \
            REGRAPH_CACHE_ROOT="${CACHE_ROOT}" \
            HF_HOME="${HF_HOME}" \
            HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" \
            HF_HUB_CACHE="${HF_HUB_CACHE}" \
            HF_ASSETS_CACHE="${HF_ASSETS_CACHE}" \
            TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}" \
            XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
            TMPDIR="${TMPDIR}" \
            REGRAPH_TOKENIZER_PATH="${REPO_ROOT}/Tokenizer/output_tokenizer/llvm_ir_bpe.json" \
            python3 "${DRIVER}" "${driver_args[@]}"
submit_status=$?
set -e

set +e
ray_status_output=$(run_in_container_on_node "${head_node}" \
    ray job status \
        --address="${ray_job_address}" \
        --log-style=record \
        --log-color=false \
        "${ray_job_id}" 2>&1)
status_status=$?
set -e

printf '%s\n' "${ray_status_output}"

if [ "${submit_status}" -ne 0 ] || [ "${status_status}" -ne 0 ]; then
    log "ERROR: Ray job submit/status failed submit_status=${submit_status} status_status=${status_status}"
    exit 1
fi

if grep -Eiq "SUCCEEDED|succeeded" <<<"${ray_status_output}"; then
    log "job finished successfully"
else
    log "ERROR: Ray job did not succeed"
    run_in_container_on_node "${head_node}" ray job logs --address="${ray_job_address}" "${ray_job_id}" || true
    exit 1
fi
