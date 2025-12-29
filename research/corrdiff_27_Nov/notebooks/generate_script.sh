#!/bin/bash
# filepath: /home/lep-zerjs/leap/DB_scratch/physicsnemo/research/corrdiff_27_Nov/scripts/run_script_nyc_generate.sh
#SBATCH --job-name=nyc-generate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --qos=leap
#SBATCH --time=24:00:00

# User inputs
YEAR=${YEAR:-2012}
NUM_ENSEMBLES=${NUM_ENSEMBLES:-10}
LEN_TO_GEN=${LEN_TO_GEN:-}
DEVICE=${DEVICE:-cuda}

CONTAINER=/home/lep-zerjs/leap/mike_testing/physicsnemo-2511-corrdiff.simg
SCRIPT=/leap/DB_scratch/physicsnemo/research/corrdiff_27_Nov/notebooks/generate_long.py
DATA_PATH=/leap/NYC_data_128/4_dec/
FILE_PATH=ERA5_hrrr_interp_128_1945_to_2005.nc
OUT_DIR=/leap/NYC_data_128/generated_1945_2005

echo "Running year=$YEAR, ensembles=$NUM_ENSEMBLES, len_to_gen=${LEN_TO_GEN:-None}, device=$DEVICE"

srun apptainer exec --bind /home/lep-zerjs/leap:/leap --nv "$CONTAINER" \
  python "$SCRIPT" \
    --year "$YEAR" \
    --num-ensembles "$NUM_ENSEMBLES" \
    ${LEN_TO_GEN:+--len-to-gen "$LEN_TO_GEN"} \
    --device "$DEVICE" \
    --data-path "$DATA_PATH" \
    --file-path "$FILE_PATH" \
    --out-dir "$OUT_DIR"