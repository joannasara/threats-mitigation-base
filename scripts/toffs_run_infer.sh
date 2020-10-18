if [[ "$(pwd)" == *threats-mitigation-base-v21 ]]
then
	if [ -z "$1" ]
	then
		docker run --gpus all \
			-v /$(pwd):/usr/toffs_infer \
			-u $(id -u ${USER}):$(id -g ${USER}) \
			-it toffs_infer:1.0 "src/config.yaml" "$1" "$2" "$3"
	else
		docker run --gpus all \
			-v /$(pwd):/usr/toffs_infer \
			-u $(id -u ${USER}):$(id -g ${USER}) \
			-it toffs_infer:1.0 "src/config.yaml" "$1" "$2" "$3"
	fi
else
	echo "please run this script from threats-mitigation-base root folder"
fi
