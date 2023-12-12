import os
import time
import torch
import torchaudio
import multiprocessing
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def load_model(config_path, checkpoint_dir):
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir)
    model.cuda()
    return model

def inference_process(model_info, text, n_times, process_id, result_queue):
    model = load_model(*model_info)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["female.wav"])

    for i in range(n_times):
        start_time = time.time()
        # Log start time with process ID and inference iteration
        print(f"Process {process_id}, Inference {i+1}: Start at {start_time}")
        out = model.batch_inference(
            text,
            "en",
            gpt_cond_latent,
            speaker_embedding,
            temperature=0.7,
        )
        end_time = time.time()
        # Log end time with process ID and inference iteration
        print(f"Process {process_id}, Inference {i+1}: End at {end_time}")
        result_queue.put(end_time - start_time)

# Number of instances to load and number of inferences
num_instances = 1 # Set the desired number of instances
n_inferences = 3  # Set the desired number of inferences per instance

config_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
checkpoint_dir = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"

text = "It took me quite a long time to develop a voice and now that I have it I am not going to be silent."

# Set up multiprocessing
processes = []
result_queue = multiprocessing.Queue()

# Start the processes
for i in range(num_instances):
    p = multiprocessing.Process(target=inference_process, args=((config_path, checkpoint_dir), text, n_inferences, i, result_queue))
    processes.append(p)
    p.start()

# Wait for all processes to complete
for p in processes:
    p.join()

total_inference_time = 0
total_inferences = num_instances * n_inferences

# Retrieve and print results
while not result_queue.empty():
    x = result_queue.get()
    print(f"Inference Time: {x} seconds")
    total_inference_time+=x


average_inference_time = total_inference_time / total_inferences
print(f"Average Inference Time: {average_inference_time} seconds")

# GPU Utilization (to be monitored externally, e.g., via nvidia-smi)

# import multiprocessing.pool
# import time
# import torch
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts

# def load_model(config_path, checkpoint_dir):
#     config = XttsConfig()
#     config.load_json(config_path)
#     model = Xtts.init_from_config(config)
#     model.load_checkpoint(config, checkpoint_dir=checkpoint_dir)
#     model.eval()  # Set the model to evaluation mode
#     return model

# def inference_task(task_data):
#     config_path, checkpoint_dir, text, process_id, n_inferences = task_data
#     model = load_model(config_path, checkpoint_dir)

#     gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["female.wav"])

#     for i in range(n_inferences):
#         start_time = time.time()
#         with torch.no_grad():  # Ensure no gradient computation
#             out = model.inference(
#                 text,
#                 "en",
#                 gpt_cond_latent,
#                 speaker_embedding,
#                 temperature=0.7,
#             )
#         end_time = time.time() - start_time
#         print(f"Thread {process_id}, Inference {i + 1}: Duration {end_time} seconds")

# if __name__ == "__main__":
#     num_instances = 5
#     n_inferences = 25  # Number of inferences per instance

#     text = "It took me quite a long time to develop a voice and now that I have it I am not going to be silent."
#     config_path = "/path/to/XTTS/config.json"
#     checkpoint_dir = "/path/to/XTTS/checkpoint/"

#     task_data = [(config_path, checkpoint_dir, text, i, n_inferences) for i in range(num_instances)]

#     with multiprocessing.pool.ThreadPool(num_instances) as pool:
#         pool.map(inference_task, task_data)