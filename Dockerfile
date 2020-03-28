FROM cloudsseas/anaconda_thesis_env:v1.0
RUN apt-get update
RUN apt-get -y install nginx

#centos 通过yum安装nginx在usr/local ,ubuntu 通过apt-get安装,在usr/share
COPY nginx.conf /usr/share/nginx/conf/html

COPY www /app
WORKDIR /app
EXPOSE 5000
ENTRYPOINT ["entrypoint.sh"]
