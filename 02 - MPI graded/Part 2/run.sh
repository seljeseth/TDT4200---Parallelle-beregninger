#IMG1="man9"
#IMG2="man10"
IMG1="woman1"
IMG2="woman5"
ROOTDIR="."
IMG_PATH="./images"
LINE_PATH="./lines/lines-${IMG1}-${IMG2}"
OUTPUT_PATH="./output-png/"
STEPS=90
CORES=2

make
rm ${OUTPUT_PATH}*.png
cmd="mpirun -np ${CORES} ${ROOTDIR}/main ${IMG_PATH}/${IMG1}.jpg ${IMG_PATH}/${IMG2}.jpg ${OUTPUT_PATH} ${STEPS} ${LINE_PATH}.txt"
echo $cmd
eval "$cmd"

./generate_video.sh
