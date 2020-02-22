#!/usr/bin/env bash
python track_pf.py --should_plot \
                   --extract_density \
                   --plot_detected_mean \
                   --resample_mode systematic \
                   --num_particles 10000 \
                   yolo_tiny