echo "System info"
echo -e "\n *** GPU Matrx ***\n"
nvidia-smi topo --matrix
echo -e "\n *** CPU Info ***\n"
lscpu
echo -e "\n *** Mem Info ***\n"
lsmem
echo -e "\n *** GPU Info ***\n"
nvidia-smi -q
echo -e "\n *** NVlink Info ***\n"
nvidia-smi nvlink --capabilities
echo -e "\n *** Conda Info ***\n"
conda info
echo -e "\n *** Conda Info ***\n"
nvcc --version
whereis cuda
whereis cudnn.h
cat $(whereis cudnn.h) | grep CUDNN_MAJOR -A 2
cat $(whereis cuda)/include/cudnn.h | grep CUDNN_MAJOR -A 2
python -c "import torch; print('Pytorch Ver:', torch.__version__)"
