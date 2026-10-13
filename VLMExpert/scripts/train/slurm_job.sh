#!/usr/bin/env bash
#SBATCH --job-name=psix
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=.logs/slurm-train-%j.out
#SBATCH --error=.logs/slurm-train-%j.err

PROJECT_ROOT=$(pwd)
TRAIN_CMD="$*"

mkdir -p "$PROJECT_ROOT/.logs"

# Keep JIT/compiler scratch and cache off BeeGFS. Each node sees its own /tmp.
export H100_JOB_CACHE_ROOT="/tmp/h100-job-cache/${USER:-songlinwei}/${SLURM_JOB_ID}"
export TRITON_HOME="$H100_JOB_CACHE_ROOT/triton-home"
export TRITON_CACHE_DIR="$H100_JOB_CACHE_ROOT/triton-cache"
export TRITON_OVERRIDE_DIR="$H100_JOB_CACHE_ROOT/triton-override"
export TRITON_DUMP_DIR="$H100_JOB_CACHE_ROOT/triton-dump"
export TORCHINDUCTOR_CACHE_DIR="$H100_JOB_CACHE_ROOT/torchinductor"
export CUDA_CACHE_PATH="$H100_JOB_CACHE_ROOT/nv"
export TORCH_EXTENSIONS_DIR="$H100_JOB_CACHE_ROOT/torch_extensions"
export XDG_CACHE_HOME="$H100_JOB_CACHE_ROOT/xdg-cache"

# Enroot runtime/data/cache paths are managed by /etc/enroot/enroot.conf.d/10-h100-defaults.conf.
# Keep them node-local to avoid BeeGFS metadata pressure during Pyxis startup.
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(getent ahostsv4 "$MASTER_NODE" | awk '{print $1; exit}')
MASTER_ADDR=${MASTER_ADDR:-$MASTER_NODE}
MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))
NNODES=$SLURM_NNODES

echo "Job ID:      $SLURM_JOB_ID"
echo "Nodes:       $SLURM_JOB_NODELIST"
echo "Master:      $MASTER_ADDR:$MASTER_PORT"
echo "Num nodes:   $NNODES"
echo "Train cmd:   $TRAIN_CMD"


# NCCL config — aligned with tested 48-GPU all-reduce reference
# bond0.1417 is the VLAN-tagged IB interface; bond0 alone misses it
export NCCL_SOCKET_IFNAME=ibp24s0,ibp25s0,ibp66s0,ibs5,ibs7,ibs8,ibs10,ibs11
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9,mlx5_10
export NCCL_DEBUG=INFO
export NCCL_RAS_ENABLE=0
export NCCL_NVLS_ENABLE=0
export NCCL_MNNVL_ENABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
# take everything offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# export WANDB_MODE=offline

# Pin the wandb run id to the Slurm job id: scontrol requeue keeps the same
# SLURM_JOB_ID, so the restarted job attaches to the SAME wandb run instead of
# opening a new one. base.py only sets WANDB_RUN_ID if it is not already set.
export WANDB_RUN_ID="slurm-${SLURM_JOB_ID}"
export WANDB_RESUME=allow

# --- Auto-requeue on walltime -----------------------------------------------
# `#SBATCH --signal=B:USR1@120` makes Slurm send SIGUSR1 to THIS batch shell (the
# `B:` prefix targets the batch step, not srun/training) 120s before the walltime
# kill. Trap it and requeue the SAME job (`scontrol requeue $SLURM_JOB_ID`) so it
# restarts with the identical submission — same sbatch flags and same "$@" (e.g.
# `--resume <ts>`) — and auto-resumes from the latest checkpoint. `#SBATCH --requeue`
# permits it; `#SBATCH --open-mode=append` keeps the pre-requeue log.
_REQUEUED=0
_requeue_on_walltime() {
    echo ">>> $(date '+%H:%M:%S') SIGUSR1: nearing walltime -> scontrol requeue ${SLURM_JOB_ID}"
    if scontrol requeue "${SLURM_JOB_ID}"; then
        _REQUEUED=1
    else
        echo ">>> $(date '+%H:%M:%S') WARNING: scontrol requeue failed; job will end at walltime without restart." >&2
    fi
}
trap _requeue_on_walltime USR1

srun_exit=0
srun \
  --ntasks=$SLURM_NNODES \
  --ntasks-per-node=1 \
  --container-image=/mnt/beegfs/containers/nvidia-pytorch-25.06-py3-cuda12.9.sqsh \
  --container-mounts=/mnt/beegfs:/mnt/beegfs \
  --container-workdir="$PROJECT_ROOT" \
  bash -lc "
    export TMPDIR=\"$H100_JOB_CACHE_ROOT/tmp\"
    export TMP=\"\$TMPDIR\"
    export TEMP=\"\$TMPDIR\"
    export TORCH_EXTENSIONS_DIR=\"$H100_JOB_CACHE_ROOT/torch_extensions\"
    export XDG_CACHE_HOME=\"$H100_JOB_CACHE_ROOT/xdg-cache\"
    mkdir -p \"\$TMPDIR\" \"$TRITON_HOME\" \"$TRITON_CACHE_DIR\" \"$TRITON_OVERRIDE_DIR\" \"$TRITON_DUMP_DIR\" \"$TORCHINDUCTOR_CACHE_DIR\" \"$CUDA_CACHE_PATH\" \"$TORCH_EXTENSIONS_DIR\" \"$XDG_CACHE_HOME\"
    export OMP_NUM_THREADS=16
    export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
    export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
    export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:\${LD_LIBRARY_PATH:-}
    export FFMPEG_DIR=/mnt/beegfs/shared/miniforge3/envs/py311
    export PATH=\$FFMPEG_DIR/bin:\$PATH
    export IMAGEIO_FFMPEG_EXE=\$FFMPEG_DIR/bin/ffmpeg
    export LD_LIBRARY_PATH=\$FFMPEG_DIR/lib:\$LD_LIBRARY_PATH
    export NNODES=$NNODES
    export NODE_RANK=\$SLURM_PROCID
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    bash $TRAIN_CMD
" &
srun_pid=$!

# Run srun in the BACKGROUND and `wait` on it (not foreground): bash defers a trap
# until the current *foreground* command returns, so a foreground srun would only fire
# _requeue_on_walltime at the hard walltime kill — too late to requeue. `wait` is
# signal-interruptible, so SIGUSR1 runs the trap at once while the job is still RUNNING.
wait "$srun_pid"
srun_exit=$?

if [[ "$_REQUEUED" == "1" ]]; then
    # Requeue is scheduled; block until Slurm tears the step down (so we don't exit into a
    # completing/requeued race), then exit 0 -> job returns to the queue and restarts,
    # auto-resuming from the latest checkpoint via the unchanged "$@".
    wait "$srun_pid" 2>/dev/null || true
    echo ">>> $(date '+%H:%M:%S') walltime reached: job requeued (JobID $SLURM_JOB_ID); will auto-resume from checkpoint on restart."
    exit 0
fi

if [[ $srun_exit -ne 0 ]]; then
    echo "srun exited with code $srun_exit"
    exit $srun_exit
fi
