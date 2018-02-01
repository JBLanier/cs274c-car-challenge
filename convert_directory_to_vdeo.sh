
#This script is totally broken, DO NOT RUN THIS, just use as example.


file="$1"
cp -r "$folder" convert_temp_copy
cd convert_temp_copy

#These are the two important commands (they work):
#Rename all files in current dir to sequential numbers -starting at 1-
ls | cat -n | while read n f; do mv "$f" "$n.jpg"; done
#Encode renamed files into 20fps mp4
ffmpeg -r 20 -f image2 -s 640x480 -i %d.jpg -vcodec libx264 -crf 15 -pix_fmt yuv420p video.mp4


# rm -r convert_temp_copy