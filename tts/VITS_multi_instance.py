import torch
import psutil
import threading
import torchaudio
from transformers import VitsTokenizer, VitsModel, set_seed
import time
import multiprocessing

def load_model(model_name):
    #device ='cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VitsModel.from_pretrained(model_name).to(device)

    text = "Loaded model successfully"
    return model, device, text


def print_cpu_info():
    print("CPU Information:")
    print(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"Total cores/threads: {psutil.cpu_count(logical=True)}")
    print(f"Max Frequency: {psutil.cpu_freq().max:.2f}Mhz")
    print(f"Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    return 1


def monitor_cpu_usage(interval=1):
    """ Monitor and print CPU usage per core at specified intervals """
    try:
        while True:
            print(f"CPU Usage per core: {psutil.cpu_percent(interval=interval, percpu=True)}%")
    except KeyboardInterrupt:
        print("Stopped CPU monitoring")

def inference_process(model_name, text, n_times, process_id, result_queue):
    model, device,text  = load_model(model_name)

    print(text)

    tokenizer = VitsTokenizer.from_pretrained(model_name)
    
    print("Tokenizing")
    inputs = tokenizer(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
    set_seed(555)  # make deterministic

    for i in range(n_times):
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time() - start_time
        result_queue.put(end_time)
        print(f"Process {process_id}, Inference {i+1}: Duration {end_time} seconds")

# cpu_monitor_thread = threading.Thread(target=monitor_cpu_usage)
# cpu_monitor_thread.start()

# Number of instances to load and number of inferences
num_instances = 3 # Set the desired number of instances
n_inferences = 25  # Set the desired number of inferences per instance

text = "The sun sets behind the mountains, painting the sky in hues of orange and pink."
model_name = "facebook/mms-tts-eng"

# Set up multiprocessing
processes = []
result_queue = multiprocessing.Queue()

# Start the processes
for i in range(num_instances):
    p = multiprocessing.Process(target=inference_process, args=(model_name, text, n_inferences, i, result_queue))
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
