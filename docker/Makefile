all:
	echo "0"

build:
	# Создание образа
	sudo docker build -t="zuev/neurolab" .

create:
	# Создание и запуск контейнера в интерактивном режиме
	sudo docker run -it --name neurolab -v /home/user/dev/neurolab:/neurolab zuev/neurolab cd neurolab
start:
    # Запуск контейнера в интерактивном режиме
	sudo docker start -i neurolab

