#!/bin/bash
#SBATCH --job-name=test_ida_container
#SBATCH --partition=workq
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/zhoul0e/ReGraphv2/Scripts/ray_opt_ablation/slurm_logs/test_ida_container-%j.out
#SBATCH --error=/scratch/zhoul0e/ReGraphv2/Scripts/ray_opt_ablation/slurm_logs/test_ida_container-%j.err

set -euo pipefail

IMAGE="/scratch/zhoul0e/regraph-data-env-llvm18.1.3.sif"
RELL_PYTHON="/scratch/zhoul0e/miniconda3/envs/ReLL/bin/python"
IDA2LLVM="/scratch/zhoul0e/ReGraphv2/Scripts/ida2llvm.py"
IDADIR="/scratch/zhoul0e/ida-pro-9.3"
HOME_DIR="/scratch/zhoul0e"
TEST_BINARY="/scratch/zhoul0e/ReGraphv2/Binaries/Dataset-1/openssl/mips32-gcc-4.8-O0_afalg.so"
OUTPUT_LL="/tmp/test_ida_${SLURM_JOB_ID}.ll"

echo "=== Test: IDA Pro inside Singularity Container ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Test binary: ${TEST_BINARY}"
echo "Binary size: $(stat -c%s "${TEST_BINARY}") bytes"
echo "Output: ${OUTPUT_LL}"
echo ""

echo "=== Loading Singularity ==="
module load singularity 2>&1
echo "Singularity version: $(singularity --version)"
echo ""

echo "=== Checking files from inside container ==="
singularity exec --bind /scratch/zhoul0e:/scratch/zhoul0e "${IMAGE}" bash -c "
echo 'Container hostname:' \$(hostname)
echo 'ReLL Python exists:' \$([ -f '${RELL_PYTHON}' ] && echo YES || echo NO)
echo 'ida2llvm.py exists:' \$([ -f '${IDA2LLVM}' ] && echo YES || echo NO)
echo 'IDADIR exists:' \$([ -d '${IDADIR}' ] && echo YES || echo NO)
echo 'Test binary exists:' \$([ -f '${TEST_BINARY}' ] && echo YES || echo NO)
echo ''

echo '=== Checking ReLL Python from container ==='
'${RELL_PYTHON}' --version 2>&1 || echo 'FAILED: Cannot run ReLL Python from container'

echo ''
echo '=== Checking IDA dependencies from container ==='
ldd '${IDADIR}/idalib.so' 2>&1 || echo 'ldd failed (may be OK)'
ldd '${IDADIR}/idat64' 2>&1 || echo 'ldd on idat64 failed (may be OK)'
" 2>&1

echo ""
echo "=== Attempting IDA lift from inside container ==="
singularity exec --bind /scratch/zhoul0e:/scratch/zhoul0e "${IMAGE}" env \
    HOME="${HOME_DIR}" \
    IDADIR="${IDADIR}" \
    REGRAPH_IDA_HOME="${HOME_DIR}" \
    bash -c "
export HOME='${HOME_DIR}'
export IDADIR='${IDADIR}'
export REGRAPH_IDA_HOME='${HOME_DIR}'
echo 'Running: ${RELL_PYTHON} ${IDA2LLVM} -f ${TEST_BINARY} -o ${OUTPUT_LL}'
'${RELL_PYTHON}' '${IDA2LLVM}' -f '${TEST_BINARY}' -o '${OUTPUT_LL}' -v 2>&1
EXIT_CODE=\$?
echo ''
echo 'Exit code:' \$EXIT_CODE
if [ \$EXIT_CODE -eq 0 ] && [ -f '${OUTPUT_LL}' ] && [ -s '${OUTPUT_LL}' ]; then
    echo 'SUCCESS: IDA lift works inside container!'
    echo 'Output size:' \$(stat -c%s '${OUTPUT_LL}') 'bytes'
    head -20 '${OUTPUT_LL}'
else
    echo 'FAILED: IDA lift did not work inside container'
    if [ -f '${OUTPUT_LL}' ]; then
        echo 'Output file exists but may be empty or wrong'
        stat '${OUTPUT_LL}' 2>&1 || true
    fi
fi
" 2>&1

echo ""
echo "=== Test complete ==="
