if [[ "$(pwd)" == *scripts ]]
then
	docker build --build-arg USER_ID=$(id -u ${USER}) --build-arg GROUP_ID=$(id -g ${USER}) -t toffs_infer:1.0 -f ../config/toffs_infer.Dockerfile ../
else
	echo "please run script from /scripts folder"
fi
