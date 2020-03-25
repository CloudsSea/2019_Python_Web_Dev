#!/usr/bin/env bash
tarfile='dist-awesome.tar.gz'
tarloc="/opt/install/docker/dataanalysis/project/dist/$tarfile"
deploydir='/srv/awesome'
git pull
echo $tarfile
echo $tarloc
echo $deploydir

sh build.sh

cd $deploydir
time=$(date "+%Y%m%d%H%M")
newdir="www$time"
echo $newdir
mkdir $newdir

cd $newdir
tar -xzvf $tarloc

#重置软链接
cd $deploydir
rm -rf www
ln -s $newdir www
#supervisord -c /etc/supervisor/supervisord.conf
#cd /etc/supervisor
#重启Python服务和nginx服务器:
#supervisorctl stop awesome
#supervisorctl start awesome
cd /usr/local/nginx/sbin
./nginx -s reload
