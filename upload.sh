#!/bin/bash

rm -rf __pycache__
cd ./test
rm -f pan_genome.fsa* log.* test.fsa_* *.npz __pycache__
cd ../

echo $PWD
#exit 0

git config --global user.email xiao.hu1@montana.edu
git config --global user.name Rinoahu


git remote rm origin

git add -A .
git commit -m 'fixed the way to import jitclass'
git remote add origin https://github.com/Rinoahu/pangenome

git pull origin master
git push origin master

git checkout master
