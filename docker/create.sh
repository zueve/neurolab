sudo docker build -t="zuev/neurolab" .

sudo docker run -it --name neurolab -v /home/user/dev/neurolab:/neurolab zuev/neurolab

# use: sudo docker start -i neurolab
