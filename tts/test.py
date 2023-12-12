import multiprocessing.pool
from transformers import VitsTokenizer, VitsModel, set_seed
import torch
import time
import argparse
from datetime import datetime
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def load_model_XTTS(config_path, checkpoint_dir):
    start_time = time.time()
    config = XttsConfig()
    config.load_json(config_path)
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=checkpoint_dir)
    model.cuda()
    load_time = time.time() - start_time
    return model , load_time

def load_model_VITS(model_name):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VitsModel.from_pretrained(model_name).to(device)
    load_time = time.time() - start_time
    return model, device, load_time

def inference_task_XTTS(task_data):
    config_path, checkpoint_dir, text, process_id, n_inferences = task_data
    model , load_time = load_model_XTTS(config_path, checkpoint_dir)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["female.wav"])

    total_time = 0
    for i in range(n_inferences):
        start_time = time.time()
        with torch.no_grad():  # Ensure no gradient computation
            out = model.inference(
                text,
                "en",
                gpt_cond_latent,
                speaker_embedding,
                temperature=0.7,
            )
        end_time = time.time() - start_time
        total_time +=end_time
        print(f"Thread {process_id}, Inference {i + 1}: Duration {end_time} seconds")
    return (total_time / n_inferences) , load_time

def inference_task_VITS(task_data):
    model_name, text, process_id, num_inferences = task_data
    model,  device , load_time = load_model_VITS(model_name)

    tokenizer = VitsTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    set_seed(555)  # make deterministic

    total_time = 0
    for i in range(num_inferences):  # Number of inferences
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time() - start_time
        total_time += end_time
        print(f"Thread {process_id}, Inference {i + 1}: Duration {end_time} seconds")
    print("printinggggggg", total_time/ num_inferences , load_time)
    return ((total_time / num_inferences), load_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run TTS Inference')
    parser.add_argument('--model', type=str, help='Model to use [VITS | XTTS]', default= 'VITS')
    parser.add_argument('--instances', type=int, help='Number of instances to run in parallel', default =1)
    parser.add_argument('--inferences', type=int,  help='Number of inferences per instance', default = 10)
    args = parser.parse_args()

    model_name = args.model
    num_instances = args.instances
    num_inferences = args.inferences
    text = "The sun sets behind the mountains, painting the sky in hues of orange and pink."
    load_time = None
    if model_name == 'VITS':
        model = "facebook/mms-tts-eng"
        task_data = [(model, text, i, num_inferences) for i in range(num_instances)]
        with multiprocessing.pool.ThreadPool(num_instances) as pool:
            resp = pool.map(inference_task_VITS, task_data)
            times, load_time = resp[0][0] , resp[0][1]

    elif model_name == "XTTS":
        config_path = "/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json"
        checkpoint_dir ="/root/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"

        task_data = [(config_path, checkpoint_dir, text, i, num_inferences) for i in range(num_instances)]
        with multiprocessing.pool.ThreadPool(num_instances) as pool:
            resp = pool.map(inference_task_XTTS, task_data)
            times , load_time = resp[0][0] , resp[0][1]

    #avg_time = sum(times) / len(times)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("inference_times.txt", "a") as file:
        file.write(f"Run at {current_time}\n")
        file.write(f"Model name: {model_name}\n")
        file.write(f"Model load time: {load_time}\n")
        file.write(f"Number of instances: {num_instances}\n")
        file.write(f"Number of inferences per instance: {num_inferences}\n")
        file.write(f"Average inference time: {times} seconds\n")

        file.write("\n")
        

