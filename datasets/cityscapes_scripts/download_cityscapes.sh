cd || exit
mkdir data
cd data || exit
mkdir cityscapes
cd cityscapes || exit
USERNAME=$1
PASSWORD=$2
wget --keep-session-cookies --save-cookies=cookies.txt --post-data "username=$USERNAME&password=$PASSWORD&submit=Login" https://www.cityscapes-dataset.com/login/

wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3

unzip -q gtFine_trainvaltest.zip
rm README license.txt
unzip -q leftImg8bit_trainvaltest.zip
rm README license.txt
rm gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip
