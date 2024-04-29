#!/bin/bash

# Bash script to run the Python script in different GNU screen sessions

num_segments=5

# Start a new tmux session in detached mode
tmux new-session -d -s dataset_processing
# tmux new-session -d -s dataset_processing 'echo "Starting dataset processing..."'
sleep 1
# Create windows in tmux for each segment
for i in $(seq 0 $((num_segments - 1)))
do
    tmux new-window -t dataset_processing -n "segment_$i" "python transform_aitw.py $i $num_segments"
done

# Optional: Attach to the session if you want to watch it running
# tmux attach -t dataset_processing

# After the setup, display a message on how to attach to the session
echo "To attach to the tmux session and monitor the processing, use:"
echo "  tmux attach -t dataset_processing"
echo "To detach, press Ctrl+B, then D."