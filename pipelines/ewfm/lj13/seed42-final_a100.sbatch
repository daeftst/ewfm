#!/bin/bash

#SBATCH --job-name=lj13_wo_annealing_wo_iterative_seed42_final   # Job name
#SBATCH --gres=gpu:1                           # Number of GPUs
#SBATCH --mem=72G                              # Memory allocation
#SBATCH --cpus-per-task=4                      # Number of CPU cores per task
#SBATCH --time=2-12:00                         # Maximum runtime (2 days)
# #SBATCH --partition=your_partition              # Partition to submit to (adapt to your cluster)
# #SBATCH --qos=your_qos                         # Quality of service (adapt to your cluster)

# Parameters
EXPERIMENT="lj13"
ENERGY="lj13"  # Same as experiment
USE_BEST_CHECKPOINT_INITIAL=false  # Use last checkpoint of initial model
USE_BEST_CHECKPOINT_EVAL=true     # Use last checkpoint of evaluation model
NUM_SAMPLES=100000
SEED=42
ENABLE_DETAILED_TRAIN_LOGGING=true

# Create unique run identifier for this pipeline run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SAMPLES_DATASET_NAME="eval_samples/seed42_final_${TIMESTAMP}"
echo "Samples will be saved with dataset name: $SAMPLES_DATASET_NAME"

# Training parameters
MAX_EPOCHS=2500
CHECK_VAL_EVERY_N_EPOCH=100
N_TRAIN_BATCHES_PER_EPOCH=20

# Model parameters
USE_EXACT_DIVERGENCE=false
ALGORITHM="baseline"

# Algorithm-specific parameters
BAGGING_BUFFER_SIZE=5000
NUM_SAMPLES_PER_BATCH=5000

# Batch size parameters
METRIC_BATCH_SIZE=2500
SAMPLE_BATCH_SIZE=5000
TEST_BATCH_SIZE=1024
EVAL_BATCH_SIZE=1024
VAL_PLOT_BATCH_SIZE=1024
VALIDATION_UNIFORM_SAMPLES=1024
ANNEALING_EPOCHS_PER_TEMPERATURE=2

# Clipping parameters
CLIPPING_PERCENTILE=99.9
CLIPPING_METHOD="modified_energy"

# Setup environment - adapt paths to your system
mamba activate ewfmenv

cd /path/to/your/ewfm/  # Replace with your project directory path

export WANDB_API_KEY="your_wandb_api_key_here"  # Replace with your W&B API key
export WANDB_PROJECT="ewfm"
export WANDB_ENTITY="your-wandb-entity"  # Replace with your W&B entity
export WANDB_RUN_NAME="lj13-wo-annealing-wo-iterative-seed42-final"

echo "========================================="
echo "STEP 1: Training Initial Model"
echo "========================================="

# Train initial model and capture output to extract checkpoint paths
TRAIN_OUTPUT=$(python ewfm/train.py \
    experiment=$EXPERIMENT \
    energy=$ENERGY \
    seed=$SEED \
    model.enable_detailed_train_logging=$ENABLE_DETAILED_TRAIN_LOGGING \
    model.use_exact_divergence=$USE_EXACT_DIVERGENCE \
    model.algorithm=$ALGORITHM \
    model.bagging_buffer_size=$BAGGING_BUFFER_SIZE \
    model.num_samples_per_batch=$NUM_SAMPLES_PER_BATCH \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.check_val_every_n_epoch=$CHECK_VAL_EVERY_N_EPOCH \
    data.n_train_batches_per_epoch=$N_TRAIN_BATCHES_PER_EPOCH \
    model.metric_batch_size=$METRIC_BATCH_SIZE \
    model.test_batch_size=$TEST_BATCH_SIZE \
    model.eval_batch_size=$EVAL_BATCH_SIZE \
    model.val_plot_batch_size=$VAL_PLOT_BATCH_SIZE \
    model.validation_uniform_samples=$VALIDATION_UNIFORM_SAMPLES \
    model.annealing_epochs_per_temperature=$ANNEALING_EPOCHS_PER_TEMPERATURE \
    model.clipping_percentile=$CLIPPING_PERCENTILE \
    model.clipping_method=$CLIPPING_METHOD \
    model.enable_annealing=false \
    model.use_ema=false \
    2>&1)
echo "$TRAIN_OUTPUT"

# Extract checkpoint paths from training output
BEST_CKPT=$(echo "$TRAIN_OUTPUT" | grep "TRAIN_BEST_CHECKPOINT:" | sed 's/TRAIN_BEST_CHECKPOINT: //')
LAST_CKPT=$(echo "$TRAIN_OUTPUT" | grep "TRAIN_LAST_CHECKPOINT:" | sed 's/TRAIN_LAST_CHECKPOINT: //')

echo "Best checkpoint: $BEST_CKPT"
echo "Last checkpoint: $LAST_CKPT"

# Determine which checkpoint to use for initial model
if [ "$USE_BEST_CHECKPOINT_INITIAL" = "true" ] && [ -n "$BEST_CKPT" ] && [ "$BEST_CKPT" != "" ]; then
    echo "Using best validation checkpoint for initial model..."
    INITIAL_CKPT="$BEST_CKPT"
else
    echo "Using last checkpoint for initial model..."
    INITIAL_CKPT="$LAST_CKPT"
fi

echo "Selected initial checkpoint: $INITIAL_CKPT"

echo "========================================="
echo "STEP 2: Evaluating Initial Model & Generating Samples"
echo "========================================="

# Run evaluation with sample generation
python ewfm/eval.py \
    experiment=$EXPERIMENT \
    energy=$ENERGY \
    ckpt_path="$INITIAL_CKPT" \
    save_samples=true \
    num_samples=$NUM_SAMPLES \
    model.metric_batch_size=$METRIC_BATCH_SIZE \
    model.use_exact_divergence=false \
    batched_sampling=true \
    sample_batch_size=$SAMPLE_BATCH_SIZE \
    model.test_batch_size=$TEST_BATCH_SIZE \
    sample_dataset_name="$SAMPLES_DATASET_NAME"

echo "========================================="
echo "STEP 3: Training Evaluation Model on Generated Samples"
echo "========================================="

# Train evaluation model on generated samples and capture checkpoint paths
EVAL_TRAIN_OUTPUT=$(python ewfm/train.py \
    experiment=$EXPERIMENT \
    energy=$ENERGY \
    seed=$((SEED + 1)) \
    model.use_train_data=true \
    model.enable_detailed_train_logging=$ENABLE_DETAILED_TRAIN_LOGGING \
    model.use_exact_divergence=false \
    trainer.max_epochs=$MAX_EPOCHS \
    trainer.check_val_every_n_epoch=$CHECK_VAL_EVERY_N_EPOCH \
    data.n_train_batches_per_epoch=$N_TRAIN_BATCHES_PER_EPOCH \
    energy.data_path_train="${SAMPLES_DATASET_NAME}_samples.npy" \
    model.metric_batch_size=$METRIC_BATCH_SIZE \
    model.test_batch_size=$TEST_BATCH_SIZE \
    model.eval_batch_size=$EVAL_BATCH_SIZE \
    model.val_plot_batch_size=$VAL_PLOT_BATCH_SIZE \
    model.validation_uniform_samples=$VALIDATION_UNIFORM_SAMPLES \
    callbacks.model_checkpoint.monitor="val/nll" \
    callbacks.model_checkpoint.mode="min" \
    2>&1)
echo "$EVAL_TRAIN_OUTPUT"

# Extract checkpoint paths from evaluation training output
EVAL_BEST_CKPT=$(echo "$EVAL_TRAIN_OUTPUT" | grep "TRAIN_BEST_CHECKPOINT:" | sed 's/TRAIN_BEST_CHECKPOINT: //')
EVAL_LAST_CKPT=$(echo "$EVAL_TRAIN_OUTPUT" | grep "TRAIN_LAST_CHECKPOINT:" | sed 's/TRAIN_LAST_CHECKPOINT: //')

echo "Evaluation model best checkpoint: $EVAL_BEST_CKPT"
echo "Evaluation model last checkpoint: $EVAL_LAST_CKPT"

# Determine which checkpoint to use for evaluation model
if [ "$USE_BEST_CHECKPOINT_EVAL" = "true" ] && [ -n "$EVAL_BEST_CKPT" ] && [ "$EVAL_BEST_CKPT" != "" ]; then
    echo "Using best validation checkpoint for evaluation model..."
    EVAL_CKPT="$EVAL_BEST_CKPT"
else
    echo "Using last checkpoint for evaluation model..."
    EVAL_CKPT="$EVAL_LAST_CKPT"
fi

echo "Selected evaluation model checkpoint: $EVAL_CKPT"

echo "========================================="
echo "STEP 4: Evaluating Evaluation Model (ESS & NLL)"
echo "========================================="

# Redefine metric batch size to 10
METRIC_BATCH_SIZE=10

# Run final evaluation
python ewfm/eval.py \
    experiment=$EXPERIMENT \
    energy=$ENERGY \
    ckpt_path="$EVAL_CKPT" \
    save_samples=false \
    model.use_exact_divergence=true \
    model.metric_batch_size=$METRIC_BATCH_SIZE \
    model.test_batch_size=$TEST_BATCH_SIZE

echo "========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "Initial model: $INITIAL_CKPT"
echo "Evaluation model: $EVAL_CKPT"
echo "Generated samples saved to: ${SAMPLES_DATASET_NAME}_samples.npy"
echo "========================================="
