#!usr/bin/env bash
# Run format: bash -i data_pipeline.bash <Path to faked data folder to distribute> <Path to dataset root> <Path to scripts folder>
# Script will:
# 1. Setup dataset structure for ID fakes
# 2. Distribute fakes to correct directories
# 3. Save identity embeddings for ID
# 4. Convert .npy identity embeddings to .mat files

DATA_LOCATION=${1:-""}
DATASET_LOCATION=${2:-""}
SCRIPTS_LOCATION=${3:-""}

set -euo pipefail

echo "Setting up CUDA environment..."

export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=":$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"    
source ${SCRIPTS_LOCATION}/env/bin/activate

for folder in ${DATA_LOCATION}/*;
do
	NAME=$(basename ${folder})
	ID=${NAME#"ID"}
	ID_LOCATION="${DATASET_LOCATION}/ID${ID}"

	echo "Setting up dataset structure for ID${ID}..."

	mkdir -p "${ID_LOCATION}/fake/cam1/frames"
	mkdir -p "${ID_LOCATION}/fake/cam3/frames"
	mkdir -p "${ID_LOCATION}/fake/cam5/frames"

	echo "Moving over videos..."

	cp "${DATA_LOCATION}/ID${ID}/camera1.mp4" "${ID_LOCATION}/fake/cam1/camera1.mp4"
	cp "${DATA_LOCATION}/ID${ID}/camera3.mp4" "${ID_LOCATION}/fake/cam3/camera3.mp4"
	cp "${DATA_LOCATION}/ID${ID}/camera5.mp4" "${ID_LOCATION}/fake/cam5/camera5.mp4"

	echo "Converting fake videos to frames..."

	ffmpeg -i "${ID_LOCATION}/fake/cam1/camera1.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/fake/cam1/frames/frames%04d.jpg"
	echo "Fake camera 1 done."
	ffmpeg -i "${ID_LOCATION}/fake/cam3/camera3.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/fake/cam3/frames/frames%04d.jpg"
	echo "Fake camera 3 done."
	ffmpeg -i "${ID_LOCATION}/fake/cam5/camera5.mp4" -loglevel quiet -q:v 2 "${ID_LOCATION}/fake/cam5/frames/frames%04d.jpg"  
	echo "Fake camera 5 done."

	echo "Saving identity embeddings in .npy files..."
	python save_all_embeddings.py --input_dir ${ID_LOCATION}

	echo "Converting identity embeddings to .mat file..."
	python all_embedding_npy_to_mat.py --root ${ID_LOCATION} --num_cams 7 --faked_cams 1 3 5 --ID ${ID} --experiment_dir /media/eleanor/New-Volume/identity-data/mat_files

	echo "Done with ID ${ID}!"
done
