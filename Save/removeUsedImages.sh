#! /bin/bash
for i in `seq 1 1 24`;
do
	file=./used_images_person_"$i".txt
	echo "reading from "$file
	while IFS='' read -r line || [[ -n "$line" ]]; do
		if [ $line -lt 10 ]
		then
			if [ $i -lt 10 ]
			then
				rm ./0$i/frame_0000"$line"_rgb.png
				rm ./0$i/frame_0000"$line"_depth.bin
			else
				rm ./$i/frame_0000"$line"_rgb.png
				rm ./$i/frame_0000"$line"_depth.bin
			fi
		elif [ $line -lt 100 ]
		then
			if [ $i -lt 10 ]
			then
				rm ./0$i/frame_000"$line"_rgb.png
				rm ./0$i/frame_000"$line"_depth.bin
			else
				rm ./$i/frame_000"$line"_rgb.png
				rm ./$i/frame_000"$line"_depth.bin
			fi
		else
			if [ $i -lt 10 ]
			then
				rm ./0$i/frame_00"$line"_rgb.png
				rm ./0$i/frame_00"$line"_depth.bin
			else
				rm ./$i/frame_00"$line"_rgb.png
				rm ./$i/frame_00"$line"_depth.bin
			fi
		fi
	done < "$file"
done
