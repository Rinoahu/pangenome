#!/bin/bash

cd ./test
rm -f pan_genome.fsa* log.* test.fsa_* *.npz
cd ../

echo $PWD
#exit 0

git config --global user.email xiao.hu1@montana.edu
git config --global user.name Rinoahu


git remote rm origin

git add -A .
git commit -m 'seperate numba jitclass version from pure python'
git remote add origin https://github.com/Rinoahu/pangenome

git pull origin master
git push origin master

git checkout master
