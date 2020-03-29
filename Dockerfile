FROM cloudsseas/anaconda_thesis_env:v1.1
# 镜像里已经更新,且安装了
#RUN apt-get update
#RUN apt-get -y install nginx

#centos 通过yum安装nginx在usr/local ,ubuntu 通过apt-get安装,在usr/share
COPY www /app
WORKDIR /app
#COPY nginx.conf /usr/share/nginx/conf/
EXPOSE 5000
ENTRYPOINT ["/app/entrypoint.sh"]
