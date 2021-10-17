OUTPUT_PATH="./output-png/"
generate_video () {
    a=1
    for i in $( ls -1v ${OUTPUT_PATH}*.png  ); do
        new=$(printf "${OUTPUT_PATH}%03d.png" "$a")
        mv -i -- "$i" "$new"
        let a=a+1
    done
    ffmpeg -framerate 30 -i ${OUTPUT_PATH}%03d.png output.mp4
}
generate_video
