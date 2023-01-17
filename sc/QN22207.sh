#!/bin/bash


WANDB_KEY="fae11c5bf2e02ec19bbe92849e7637a568959380"
# change this too!
USERID='users/gzelenfroind'


DESIRED_GLOBAL_BATCH_SIZE=2048

DATE=`date +%m-%d`
# TAG='nightly'

# we can go to nightly once draco upgrades drivers
TAG='2204'

CLUSTER_TAG='drc'
MAX_RUNTIME='8:00:00'
SLURM_ACCOUNT='ent_aiapps'
SRUN_ACCOUNT='ent_aiapps_asr'

TORCH_CUDNN_V8_API_ENABLED_FLAG=''

if [ $# -lt 3 ]
then
        echo "use: run.sh <node type> <nnodes> <precision>"
        exit 1
fi

NODE_TYPE="$1"
NNODES="$2"
PRECISION="$3"

if [ $PRECISION == '32' ]
then
        echo "32-bit precision will be used"
else
	echo "unsupported precision $PRECISION"
        exit 1
fi

if [ $NODE_TYPE == 'dgx1' ]
then
	echo "using node type $NODE_TYPE"
	CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'
	GPUS_PER_NODE=8

elif [ $NODE_TYPE == 'dgx2h' ]
then
	echo "using node type $NODE_TYPE"
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'
        GPUS_PER_NODE=16
else
	echo "unsupported node type $NODE_TYPE"
	exit 1
fi

PARTITION="batch_${NODE_TYPE}_m2"
echo "submitting to partition ${PARTITION}"

MAX_EPOCHS=150

EXP_NAME="quartz-2-${CLUSTER_TAG}-${MAX_EPOCHS}e-${NODE_TYPE}-${PRECISION}-${NNODES}n-${TAG}-${DATE}"
# PROJECT_NAME='cnfr_convergence'
PROJECT_NAME='Quartznet20'

LOG_EVERY_N_STEPS=100
PROGRESS_BAR_REFRESH=10

# TRAIN_BATCH=16
# FUSED_TRAIN_BATCH=16
TRAIN_BATCH=8
FUSED_TRAIN_BATCH=8
MAX_AUDIO_DURATION=20

GRAD_ACCUM=$((DESIRED_GLOBAL_BATCH_SIZE / TRAIN_BATCH / NNODES / GPUS_PER_NODE))
echo "setting grad_accum to $GRAD_ACCUM to ensure global batch size of $DESIRED_GLOBAL_BATCH_SIZE"

EVAL_BATCH_SIZE=64

TOKENIZER_TYPE=bpe 
TOKENIZER="/tokenizers/librispeech/librispeech_tokenizer_spe_unigram_v1024"

OPTIM="adamw"
LR=5.0
BETAS=[0.9,0.98]
WEIGHT_DECAY=1e-3

SCHED=NoamAnnealing
WARMUP_STEPS=10000
MIN_LR=1e-6



DFS_ACCOUNT_PREFIX=/gpfs/fs1/projects/${SLURM_ACCOUNT}

# CONTAINER="nvcr.io/nvidian/ac-aiapps/nemo:${TAG}"
CONTAINER="nvcr.io/nvidia/nemo:22.07"

TRAIN_MANIFEST_FILEPATH='/data2/librispeech_sp_tarred/tarred_audio_manifest.json'
TARRED_AUDIO_FILEPATHS='/data2/librispeech_sp_tarred/audio__OP_0..511_CL_.tar'


VAL_MANIFEST='[/data/librispeech/LibriSpeech/librivox-dev-other.json,/data/librispeech/LibriSpeech/librivox-dev-clean.json,/data/librispeech/LibriSpeech/librivox-test-other.json,/data/librispeech/LibriSpeech/librivox-test-clean.json]'
# VAL_FILEPATHS='/data/LibriSpeech/eval__OP_0..1023_CL_.tar'
# VAL_FILEPATHS='/data/data/LibriSpeech/eval__OP_0..1023_CL_.tar'

# trying to save disk space
# TOP_K=5
TOP_K=3

TRAIN_WORKERS=8

VALIDATION_BATCH=32
VALIDATION_WORKERS=8

DATA_DIR=${DFS_ACCOUNT_PREFIX}/datasets/data
DATA_VAL_DIR=${DFS_ACCOUNT_PREFIX}/datasets/data/librispeech

TOKENIZERS_DIR=${DFS_ACCOUNT_PREFIX}/datasets/tokenizers
MANIFESTS_DIR=${DFS_ACCOUNT_PREFIX}/${USERID}/manifests
YAML_DIR=${DFS_ACCOUNT_PREFIX}/${USERID}/yaml
RESULTS_DIR=${DFS_ACCOUNT_PREFIX}/${USERID}/results/$PROJECT_NAME/$EXP_NAME

mkdir -p $RESULTS_DIR
OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

# MOUNTS="--container-mounts=${RESULTS_DIR}:/results,${DATA_DIR}:/data2,${DATA_VAL_DIR}:/data,${TOKENIZERS_DIR}:/tokenizers"
# MOUNTS="--container-mounts=${RESULTS_DIR}:/results,${DATA_DIR}:/data,${DATA_VAL_DIR}:/data/librispeech/LibriSpeech,${TOKENIZERS_DIR}:/tokenizers"
MOUNTS="${RESULTS_DIR}:/results,${DATA_DIR}:/data2,${YAML_DIR}:/ws,${DATA_VAL_DIR}:/data/librispeech/LibriSpeech,${TOKENIZERS_DIR}:/tokenizers"

SCRIPT_NAME='speech_to_text_ctc.py'
SCRIPT_PATH="/workspace/nemo/examples/asr/asr_ctc"

CONFIG_PATH="/ws/"
CONFIG_NAME="qn122M_k7.yaml"


read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& nvidia-smi \
&& wandb login ${WANDB_KEY} \
&& CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} $TORCH_CUDNN_V8_API_ENABLED_FLAG python -u ${SCRIPT_PATH}/${SCRIPT_NAME}  \
--config-path=${CONFIG_PATH} \
--config-name=${CONFIG_NAME} \
++model.train_ds.batch_size=${TRAIN_BATCH} \
++model.train_ds.max_duration=${MAX_AUDIO_DURATION} \
++model.train_ds.manifest_filepath=${TRAIN_MANIFEST_FILEPATH} \
++model.train_ds.is_tarred=true \
++model.train_ds.tarred_audio_filepaths=${TARRED_AUDIO_FILEPATHS} \
++model.train_ds.num_workers=${TRAIN_WORKERS} \
++model.validation_ds.manifest_filepath=${VAL_MANIFEST} \
++model.validation_ds.batch_size=${VALIDATION_BATCH} \
++model.validation_ds.num_workers=${VALIDATION_WORKERS} \
++model.joint.fused_batch_size=${FUSED_TRAIN_BATCH} \
++exp_manager.create_wandb_logger=true \
++exp_manager.wandb_logger_kwargs.project=${PROJECT_NAME} \
++exp_manager.wandb_logger_kwargs.name=${EXP_NAME} \
++exp_manager.name=${EXP_NAME} \
++exp_manager.resume_if_exists=true \
++exp_manager.resume_ignore_no_checkpoint=true \
++exp_manager.checkpoint_callback_params.save_top_k=${TOP_K} \
++exp_manager.exp_dir=/results/ \
++exp_manager.use_datetime_version=false \
++trainer.max_epochs=${MAX_EPOCHS} \
++trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
++trainer.check_val_every_n_epoch=1 \
++trainer.num_nodes=${NNODES} \
++trainer.accumulate_grad_batches=${GRAD_ACCUM} \
++trainer.devices=-1 \
++model.optim.name=${OPTIM} \
++model.optim.lr=${LR} \
++model.optim.betas=${BETAS} \
++model.optim.weight_decay=${WEIGHT_DECAY} \
++model.optim.sched.name=${SCHED} \
++model.optim.sched.min_lr=${MIN_LR} \
++model.optim.sched.warmup_steps=${WARMUP_STEPS}
EOF

# srun
# --container-image=${CONTAINER} \
# sbatch -J ${EXP_NAME} -o ${OUTFILE} -e ${ERRFILE} \
sbatch \
  -J ${EXP_NAME} \
  --account=${SRUN_ACCOUNT} \
  --partition=${PARTITION} \
  --nodes=${NNODES} \
  --time=${MAX_RUNTIME} \
  --exclusive \
  --mem=0 \
  --mail-type=FAIL \
  --gpus-per-node=${GPUS_PER_NODE} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --nv-meta=ml-model.${EXP_NAME} \
  --overcommit \
  ./srun.sh ${EXP_NAME} ${OUTFILE} ${ERRFILE} ${CONTAINER} ${MOUNTS} "${cmd}"


#   "#!/bin/sh \
#     srun -J ${EXP_NAME} -o ${OUTFILE} -e ${ERRFILE}  --container-image=${CONTAINER} ${MOUNTS} bash -c \"${cmd}\" \
#   "

#   ${MOUNTS} \
#   bash -c "${cmd}"

