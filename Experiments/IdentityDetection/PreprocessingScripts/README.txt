run sudo ln -s /usr/local/cuda/lib64 /usr to properly link cuda

OR 

export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64"
