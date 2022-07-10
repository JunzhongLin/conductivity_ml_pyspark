#! /bin/bash

# create package directory if not exsit
mkdir -p packages

: '
# install packages to a temporary directory and zip it
touch requirements.txt  # safeguard in case there are no packages
pip3 install -r requirements.txt --target ./packages

# check to see if there are any external dependencies
# if not then create an empty file to seed zip with
if [ -z "$(ls -A packages)" ]
then
    touch packages/empty.txt
fi
'
cd packages
zip -9mrv packages.zip .
mv packages.zip ..
cd ..

# remove temporary directory
rm -rf packages

# add local modules
echo '... adding all modules from local utils package'
cd app
zip -ru9 ../packages.zip dependencies -x dependencies/__pycache__/\*
exit 0

