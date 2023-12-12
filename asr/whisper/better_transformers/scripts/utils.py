from builtins import super  # https://stackoverflow.com/a/30159479
from threading import Event, Thread
import GPUtil
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time

def get_gpus_used():
    try:
        gpus = GPUtil.getGPUs()
        return gpus[0].memoryUsed if gpus else None
    except Exception as e:
        print("ERROR: ", e)
    return None

def load_pipeline(model_name: str, to_better: bool = True, use_flash: bool = False):
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch_dtype = (torch.float16 if torch.cuda.is_available() else torch.float32)
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    print("Loading model...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.eval()
    if to_better:
        model = model.to_bettertransformer()
        
    model.to(device)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device = device,
        model_kwargs={"use_flash_attention_2": use_flash},
    )
    
    


class ThreadWithReturnValue(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return = None
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()
    
    def run(self):
        try:
            used_memory_list = []
            while not self.stopped():
                time.sleep(0.03)
                if gpu_memory := get_gpus_used():
                    used_memory_list.append(gpu_memory)
            self._return = max(used_memory_list)
        except Exception as e:
            print("ERROR in thread: ", e)
        self._return = -1
    
    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self._return
        
