#!/bin/bash -xe

make modules -j
sudo make modules_install -j
sudo rmmod nvidia_uvm
