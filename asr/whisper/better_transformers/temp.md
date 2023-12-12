docker run -dit --gpus 1 --name whisper-better --network host -v /root/arun/whisper-better/:/app ai-asr

python3 -m pip install torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

python3 bench.py --filename ../audio/en_16.wav --language en --batch_sizes 2,4 --better