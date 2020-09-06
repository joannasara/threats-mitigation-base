if [[ "$(pwd)" == *scripts ]]
then
	docker build --build-arg USER_ID=$(id -u ${USER}) --build-arg GROUP_ID=$(id -g ${USER}) -t toffs_train:1.0 -f ../config/toffs_train.Dockerfile ../
else
	echo "please run this script from the /scripts folder"
fi
