#!/bin/bash

set -x
set -u # Fail on an undefined variable

#================================================================
# 1. DEFAULT_SCRIPT_PARAMS (do not change)
#================================================================

# CLUSTER is basic variable, default params depends on it
if [[ -z "${CLUSTER:-}" ]]; then echo "ERROR: CLUSTER should be set" && exit 1; fi
# CLUSTER: draco, draco_m3, selene, ngc, local

# partition
DEFAULT_PARTITION=""
if [[ $CLUSTER = "draco" ]]; then
  DEFAULT_PARTITION="batch_dgx2h_m2"
elif [[ $CLUSTER = "draco_m3" ]]; then
  DEFAULT_PARTITION="batch_dgx1_m3"
elif [[ $CLUSTER = "selene" ]]; then
  DEFAULT_PARTITION="luna"
elif [[ $CLUSTER = "ngc" ]]; then
  DEFAULT_PARTITION="dgx1v.32g.8.norm"
fi

# container
if [[ $CLUSTER = "draco" || $CLUSTER = "draco_m3" ]]; then
  # draco has old drivers, don't use anything above 22.04
  DEFAULT_CONTAINER_TAG="v1.11.0_22.04"
else
  DEFAULT_CONTAINER_TAG="v1.11.0_22.08"
fi

if [[ $CLUSTER = "ngc" ]]; then
  # ngc doesn't work with gitlab-master
  DEFAULT_CONTAINER_REGISTRY="nvcr.io/nvidian/ac-aiapps/nemo_vb"
else
  DEFAULT_CONTAINER_REGISTRY="gitlab-master.nvidia.com/vbataev/nemo_containers"
fi

# num gpus
if [[ $CLUSTER = "draco" ]]; then
  DEFAULT_GPUS_PER_NODE=16
elif [[ $CLUSTER = "draco_m3" ]]; then
  DEFAULT_GPUS_PER_NODE=8
elif [[ $CLUSTER = "selene" ]]; then
  DEFAULT_GPUS_PER_NODE=8
elif [[ $CLUSTER = "ngc" ]]; then
  DEFAULT_GPUS_PER_NODE=8
elif [[ $CLUSTER = "local" ]]; then
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    LOCAL_GPUS="all"
  else
    LOCAL_GPUS="device=${CUDA_VISIBLE_DEVICES}"
  fi
  DEFAULT_GPUS_PER_NODE=$(docker run -it --rm --gpus '"'"${LOCAL_GPUS}"'"' "${DEFAULT_CONTAINER_REGISTRY}:${DEFAULT_CONTAINER_TAG}" nvidia-smi --list-gpus | grep -c GPU)
fi

# precision
if [[ $CLUSTER = "selene" ]]; then
  DEFAULT_PRECISION="bf16"
else
  DEFAULT_PRECISION=16
fi

DEFAULT_DL_WORKERS=8
DEFAULT_NUM_NODES=1

#================================================================
# DEFAULT_SCRIPT_PARAMS:END
#================================================================

#================================================================
# 2. COMMON_SCRIPT_PARAMS (do not change)
#================================================================

# can be redefined by env variables, e.g. run `CLUSTER=local BATCH_SIZE=256 ./experiment-tpl.sh`

# where to run
# CLUSTER is basic variable, checked above
PARTITION=${PARTITION:-${DEFAULT_PARTITION}}
GPUS_PER_NODE=${GPUS_PER_NODE:-${DEFAULT_GPUS_PER_NODE}}
WANDB_KEY="fae11c5bf2e02ec19bbe92849e7637a568959380"
# logging
if [[ -z "${WANDB_KEY:-}" ]]; then echo "ERROR: WANDB_KEY should be set" && exit 1; fi

# speed-related
DL_WORKERS=${DL_WORKERS:-${DEFAULT_DL_WORKERS}} # number of workers for dataloader
PRECISION=${PRECISION:-"${DEFAULT_PRECISION}"}  # 16, 32, bf16

# container
CONTAINER_TAG=${CONTAINER_TAG:-"${DEFAULT_CONTAINER_TAG}"}
CONTAINER_REGISTRY=${CONTAINER_REGISTRY:-${DEFAULT_CONTAINER_REGISTRY}}
CONTAINER="${CONTAINER_REGISTRY}:${CONTAINER_TAG}"

# dependency / num runs
NUM_RUNS=${NUM_RUNS:-30}    # number of consecutive runs, only for slurm
AFTER_JOB=${AFTER_JOB:-} # slurm job after which run should start

# batch-size related variables
if [[ -z "${BATCH_SIZE:-}" ]]; then echo "ERROR: BATCH_SIZE should be set" && exit 1; fi
if [[ -z "${GRAD_ACCUMULATION:-}" ]]; then echo "ERROR: GRAD_ACCUMULATION should be set" && exit 1; fi
NUM_NODES=${NUM_NODES:-${DEFAULT_NUM_NODES}}

#================================================================
# COMMON_SCRIPT_PARAMS:END
#================================================================

#================================================================
# 3. SCRIPT_ENV (do not change)
#================================================================

SCRIPT_ENV="HYDRA_FULL_ERROR=1"

if [[ $PRECISION = "bf16" ]]; then
  SCRIPT_ENV="${SCRIPT_ENV} TORCH_CUDNN_V8_API_ENABLED=1"
fi

if [[ ! -z ${PL_FAULT_TOLERANT_TRAINING:-""} ]]; then
  SCRIPT_ENV="${SCRIPT_ENV} PL_FAULT_TOLERANT_TRAINING=${PL_FAULT_TOLERANT_TRAINING}"
fi

#================================================================
# SCRIPT_ENV:END
#================================================================

#================================================================
# 4. EXPERIMENT_PARAMS
#================================================================

# experiment name + project
EXP_BASE_NAME=${EXP_BASE_NAME:-"QN20-LSP"}
PROJECT=${PROJECT:-"QN-20=Exp-LS"}

# additional experiment params
NUM_EPOCHS=${NUM_EPOCHS:-10000}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
MIN_LR=${MIN_LR:-1e-6}
LOG_EVERY_N_STEPS=${LOG_EVERY_N_STEPS:-100}

SLURM_REMOTE=${SLURM_REMOTE:-false}  # execute Draco/Selene task via ssh

#================================================================
# EXPERIMENT_PARAMS:END
#================================================================

#================================================================
# 5. DIRECTORIES
#================================================================

# directories on cluster with data (<dataset>_CLUSTER_DIR) + workspace (HOME_CLUSTER_DIR)
if [[ $CLUSTER = "draco" ]]; then
  HOME_CLUSTER_DIR="/gpfs/fs1/projects/ent_aiapps/users/gzelenfroind"
  LIBRISPEECH_CLUSTER_DIR="/gpfs/fs1/projects/ent_aiapps/datasets/data/librispeech"
  LIBRISPEECH_TARRED_CLUSTER_DIR="/gpfs/fs1/projects/ent_aiapps/datasets/data/librispeech_sp_tarred"
elif [[ $CLUSTER = "draco_m3" ]]; then
  HOME_CLUSTER_DIR="/lustre/fs1/ent/aiapps/gzelenfroind"
  LIBRISPEECH_CLUSTER_DIR="/lustre/fs1/ent/aiapps/datasets/LibriSpeech"
  LIBRISPEECH_TARRED_CLUSTER_DIR="/lustre/fs1/ent/aiapps/datasets/Librispeech_SP_Tarred"
elif [[ $CLUSTER = "selene" ]]; then
  HOME_CLUSTER_DIR="/lustre/fsw/swdl/swdl-langspeech/vbataev"
  LIBRISPEECH_CLUSTER_DIR="/lustre/fsw/swdl/swdl-langspeech/datasets/data/ASR/LibriSpeech/librispeech/LibriSpeech"
  LIBRISPEECH_TARRED_CLUSTER_DIR="/lustre/fsw/swdl/swdl-langspeech/datasets/data/librispeech_sp_tarred"
elif [[ $CLUSTER = "local" ]]; then
  HOME_CLUSTER_DIR="/home/vbataev"
  LIBRISPEECH_CLUSTER_DIR="/data/LibriSpeech"
  LIBRISPEECH_TARRED_CLUSTER_DIR="/media/hdd/Datasets/LibriSpeech/97103"
elif [[ $CLUSTER = "ngc" ]]; then
  HOME_CLUSTER_DIR="exp_asr_ngc_vb" # workspace to mount
  # NGC LibriSpeech: 26109 - sp, no dev/test, only train, no 1.0; 88225 - with folders; 9367 - separate folder "LibriSpeech"
  LIBRISPEECH_CLUSTER_DIR="9367"
  LIBRISPEECH_TARRED_CLUSTER_DIR="97103"
else
  echo "Incorrect Cluster Name" && exit 1
fi

# paths to mount directories
# workspace
WORKSPACE_MNT=/ws

# data directories
LIBRISPEECH_DIR=/data/LibriSpeech/LibriSpeech
if [[ $CLUSTER = "ngc" ]]; then
  LIBRISPEECH_DIR=/data/LibriSpeech
fi
LIBRISPEECH_TARRED_DIR=/data/LibriSpeech_tarred

# directory with code
if [[ $CLUSTER = "local" ]]; then
  CODE_DIR=${WORKSPACE_MNT}/code/exprunner
else
  CODE_DIR=${WORKSPACE_MNT}/exprunner
fi

#================================================================
# DIRECTORIES:END
#================================================================

#================================================================
# 6. MANIFESTS
#================================================================

# manifests directories
LIBRISPEECH_MANIFESTS_DIR=${CODE_DIR}/manifests/librispeech

# used manifests
#TRAIN_MANIFESTS=${LIBRISPEECH_MANIFESTS_DIR}/train_clean_100.json,${LIBRISPEECH_MANIFESTS_DIR}/train_clean_360.json,${LIBRISPEECH_MANIFESTS_DIR}/train_other_500.json
TRAIN_MANIFESTS=${LIBRISPEECH_TARRED_DIR}/tarred_audio_manifest.json
TARRED_AUDIO_FILEPATHS="${LIBRISPEECH_TARRED_DIR}/audio__OP_0..511_CL_.tar"
DEV_MANIFESTS=${LIBRISPEECH_MANIFESTS_DIR}/dev_other.json
TEST_MANIFESTS=${LIBRISPEECH_MANIFESTS_DIR}/dev_clean.json,${LIBRISPEECH_MANIFESTS_DIR}/test_clean.json,${LIBRISPEECH_MANIFESTS_DIR}/test_other.json

#================================================================
# MANIFESTS:END
#================================================================

#================================================================
# 7. EXPERIMENT_DIR
#================================================================

EXP_NAME="${EXP_BASE_NAME}--${CLUSTER}-bs${BATCH_SIZE}xg${GPUS_PER_NODE}xn${NUM_NODES}"
if [[ "${GRAD_ACCUMULATION}" -gt 1 ]]; then
  EXP_NAME="${EXP_NAME}xga${GRAD_ACCUMULATION}"
fi
EXP_NAME="${EXP_NAME}--${PRECISION}"

EXP_DIR=${WORKSPACE_MNT}/exp/${EXP_NAME}

#================================================================
# EXPERIMENT_DIR:END
#================================================================

#================================================================
# 8. EXPERIMENT_COMMAND
#================================================================

TRAIN_SCRIPT="/ws/NeMo/examples/asr/asr_ctc/speech_to_text_ctc.py"
CONFIG_PATH="/ws/"
CONFIG_NAME="qn122M_k7.yaml"
OPTIM="novograd"
LR=0.05
BETAS=[0.9,0.98]
WEIGHT_DECAY=1e-3

SCHED=CosineAnnealing
WARMUP_STEPS=10000
MIN_LR=1e-6






# define command for experiment
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
&& export WANDB_API_KEY=${WANDB_KEY} \
&& mkdir -p ${EXP_DIR} \
&& cd ${EXP_DIR} \
&& ${SCRIPT_ENV} python ${TRAIN_SCRIPT} \
--config-name=${CONFIG_NAME} \
--config-path=${CONFIG_PATH} \
++model.train_ds.manifest_filepath=${TRAIN_MANIFESTS} \
++model.train_ds.is_tarred=true \
++model.train_ds.tarred_audio_filepaths=${TARRED_AUDIO_FILEPATHS} \
++model.validation_ds.manifest_filepath=[${DEV_MANIFESTS},${TEST_MANIFESTS}] \
++model.train_ds.num_workers=${DL_WORKERS} \
++model.validation_ds.num_workers=${DL_WORKERS} \
++model.train_ds.batch_size=${BATCH_SIZE} \
++model.optim.weight_decay=${WEIGHT_DECAY} \
++model.optim.name=${OPTIM} \
++model.optim.betas=${BETAS} \
++model.optim.sched.name=${SCHED} \
++model.optim.sched.min_lr=${MIN_LR} \
++model.optim.sched.warmup_steps=${WARMUP_STEPS} \
++trainer.max_epochs=${NUM_EPOCHS} \
++trainer.num_nodes=${NUM_NODES} \
++trainer.accumulate_grad_batches=${GRAD_ACCUMULATION} \
++trainer.devices=-1 \
++trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
++exp_manager.create_wandb_logger=True \
++exp_manager.wandb_logger_kwargs.name="${EXP_NAME}" \
++exp_manager.wandb_logger_kwargs.project="${PROJECT}" \
++exp_manager.wandb_logger_kwargs.resume=auto \
++exp_manager.wandb_logger_kwargs.id="${EXP_NAME}" \
++exp_manager.resume_if_exists=true \
++exp_manager.resume_ignore_no_checkpoint=true \
++exp_manager.exp_dir=${EXP_DIR} \
++exp_manager.name=${EXP_NAME}
EOF

#================================================================
# EXPERIMENT_COMMAND:END
#================================================================

#================================================================
# 9. RUN_TASK
#================================================================

EXP_DIR_CLUSTER=${HOME_CLUSTER_DIR}/exp/${EXP_NAME} # only for slurm, ignored for local/ngc tasks

REMOTE_FLAG=""
if [[ $SLURM_REMOTE = true ]]; then
  REMOTE_FLAG="--slurm-remote"
fi
unirun \
  --name "${EXP_NAME}" \
  --wrap-name \
  --cluster "${CLUSTER}" \
  --container "${CONTAINER}" \
  --dataset "${LIBRISPEECH_CLUSTER_DIR}:${LIBRISPEECH_DIR}" \
  --dataset "${LIBRISPEECH_TARRED_CLUSTER_DIR}:${LIBRISPEECH_TARRED_DIR}" \
  --workspace "${HOME_CLUSTER_DIR}:${WORKSPACE_MNT}" \
  --cmd "${cmd}" \
  --num-runs "${NUM_RUNS}" \
  --num-nodes "${NUM_NODES}" \
  --gpus-per-node "${GPUS_PER_NODE}" \
  --partition "${PARTITION}" \
  --time-limit ${TIME_LIMIT:-"230:00"} \
  --slurm-logs-dir "${EXP_DIR_CLUSTER}" \
  --after-job "${AFTER_JOB}" \
  --save-cmd-to=file.txt \
  ${REMOTE_FLAG} --verbose

#================================================================
# RUN_TASK:END
#================================================================

set + # clean up
