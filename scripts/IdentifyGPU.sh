#!/bin/bash

lspci | grep -m 1 "controller: NVIDIA" | awk 'BEGIN { FS = "NVIDIA Corporation" } { print $2 }' | awk '{print $1}'
