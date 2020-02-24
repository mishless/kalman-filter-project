#!/usr/bin/env bash
python3 track_pf.py --extract_density \
                    --resample_mode systematic \
                   --num_particles 10000 \
                    --point_estimate \
                    yolo_tiny

python3 track_pf.py --extract_density \
                    --resample_mode systematic \
                    --num_particles 10000 \
                    yolo_tiny
