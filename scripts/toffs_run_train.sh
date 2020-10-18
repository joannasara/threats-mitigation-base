if [[ "$(pwd)" == *threats-mitigation-base-v21 ]]
then
docker run --gpus all \
	-v /$(pwd):/usr/toffs_train \
	-u $(id -u ${USER}):$(id -g ${USER}) \
	-it toffs_train:1.0 "src/config.yaml" "$1" "$2"
else
	echo "please run this script from threats-mitigation-base root folder"
fi
