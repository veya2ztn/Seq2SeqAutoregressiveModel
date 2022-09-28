
# echo `ls ~/cephdrive/`
# echo `ls ~/cephdrive/test`
# fusermount -u ~/cephdrive/test
# ~/bin/s3fs era5npy ~/cephdrive/test -o passwd_file=~/cephdrive/zhangtianning.passwd-s3fs -o url=http://10.140.2.254:80  -o use_path_request_style
# echo `ls ~/cephdrive/test/`
# echo `pwd`
python downsamplefrom720.py


sensesync --dryrun --include "1979" sync s3://FCM1NI7IC4S78J2EGJUC:zRQ5lOMyaXWdEjbqt24rcIQD9wZilMhn9v45wbPo@era5npy.10.140.2.254:80/era5npy/ ./