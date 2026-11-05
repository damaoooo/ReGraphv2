# Site settings for the Scripts/shaheen job scripts (sourced, not executed). Export these before sbatch / in your
# shell profile; nothing here is specific to one user or machine.
#   REGRAPH_REPO     this repository's checkout (sbatch runs a spooled copy of the script, so it cannot find it alone)
#   REGRAPH_SIF      the LLVM 18 data-environment Singularity image the pipeline runs in
#   REGRAPH_BIND     directories to bind into the SIF, comma separated (at least the data root and the repository)
#   IDA_TAR          IDA Pro 9.4 install as .tar.zst (unpacked to node-local /tmp per job)          [IDA stages]
#   IDAUSR_TEMPLATE  an IDAUSR directory (license, EULA accepted, ida.reg with Lumina/telemetry off) [IDA stages]
#   REGRAPH_DRIVER   batch driver for the IDA stages (default: Scripts/shaheen/ida_driver.py)
# Job logs go to ./logs/ of the submitting directory (create it first): sbatch --output=logs/%x-%A_%a.out.
: "${REGRAPH_REPO:?set REGRAPH_REPO to the ReGraphv2 checkout (see Scripts/shaheen/cluster_env.sh)}"
: "${REGRAPH_SIF:?set REGRAPH_SIF to the LLVM 18 data-environment .sif (see Scripts/shaheen/cluster_env.sh)}"
: "${REGRAPH_BIND:?set REGRAPH_BIND to the directories to bind into the SIF (see Scripts/shaheen/cluster_env.sh)}"
REGRAPH_DRIVER=${REGRAPH_DRIVER:-$REGRAPH_REPO/Scripts/shaheen/ida_driver.py}
SINGULARITY_BINDS="--bind /tmp:/tmp"
IFS=',' read -ra _binds <<< "$REGRAPH_BIND"
for _b in "${_binds[@]}"; do SINGULARITY_BINDS="$SINGULARITY_BINDS --bind $_b:$_b"; done
unset _binds _b
module load singularity >/dev/null 2>&1 || true
