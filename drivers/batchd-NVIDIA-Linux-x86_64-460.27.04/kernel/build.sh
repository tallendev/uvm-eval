#!/bin/bash -xe

make modules
sudo make modules_install
sudo rmmod nvidia_uvm
