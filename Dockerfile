FROM cloudsseas/anaconda_thesis_env:v1.0
RUN apt-get update
RUN apt-get -y install nginx
COPY nginx.conf /usr/local/nginx/conf/html

COPY www /app
WORKDIR /app
EXPOSE 5000
ENTRYPOINT ["entrypoint.sh"]
