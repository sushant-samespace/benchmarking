import argparse
import json
import time

import numpy as np
import torch
from utils import load_pipeline, get_gpus_used, ThreadWithReturnValue


parser = argparse.ArgumentParser(description="Benchmark the speech recognition model")
parser.add_argument(
    "-f",
    "--filename",
    type=str,
    # default="./samples/jfk.wav",
    help="Relative path of the file to transcribe (default: ./samples/jfk.wav)",
)

parser.add_argument(
    "--language",
    required=False,
    type=str,
    default="en",
    help='Language of the input audio. (default: "None" (Whisper auto-detects the language))',
)

parser.add_argument(
    "-b",
    "--batch_sizes",
    type=str,
    default="4",
    help="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 4)",
)

parser.add_argument(
    "-fl",
    "--flash",
    action='store_true',
    help="Use Flash Attention 2. Read the FAQs to see how to install FA2 correctly. (default: False)",
)

parser.add_argument(
    "-bt",
    "--better",
    action='store_true',
    help='Use Better Transformer',
)

if __name__ == "__main__":

    args = parser.parse_args()
    print("args: ", args)

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    # load audio
    audio_bytes = open(args.filename, "rb").read()
    audio_bytes_array = np.frombuffer(audio_bytes, np.int16).flatten().astype(np.float32) / 32768.0
    
    models = ['openai/whisper-large-v3', 'openai/whisper-large-v2', 'openai/whisper-medium']
    models = ['openai/whisper-tiny'] 

    benchmark_results = []

    for model_name in models:
        # Load
        gpu_before_load = get_gpus_used()
        start_load_time = time.time()
        asr_pipeline = load_pipeline(
            model_name=model_name,
            to_better=args.better,
            use_flash=args.flash,
        )
        load_time = time.time() - start_load_time
        time.sleep(1)
        gpu_after_load = get_gpus_used()
        
        # Warmup the model
        print("Warming up the model...")
        for _ in range(2):
            _ = asr_pipeline([audio_bytes_array for _ in range(3)], generate_kwargs={"task": "transcribe", "language": args.language, "num_beams": 5}, batch_size=3)
        
        print("gpu_after_load: ", gpu_after_load)
        for batch_size in batch_sizes:
            
            audio_batch = [audio_bytes_array for _ in range(batch_size)]
            
            # infer Model
            print("Start thread...")
            thread = ThreadWithReturnValue()
            thread.start()
    
            print("Trigger inference...")
            start_inference_time = time.time()
            resp = asr_pipeline(audio_batch, generate_kwargs={"task": "transcribe", "language": args.language, "num_beams": 5}, batch_size=batch_size)
            inference_time = time.time() - start_inference_time
            print("Done inference...")
            
            thread.stop()
            time.sleep(2)
            max_gpu_used = thread.join() #thread._return
            
            benchmark_results.append(
                {
                    "model": model_name,
                    "language": args.language,
                    "better_transformers": args.better,
                    "flash_attention_v2": args.flash,
                    "batch_size": batch_size,
                    "load_time": load_time,
                    "gpu_before_load": gpu_before_load,
                    "gpu_after_load": gpu_after_load,
                    "inference_time": inference_time,
                    "max_gpu_used": max_gpu_used,
                    "response": resp
                }
            )
            
            json.dump(benchmark_results, open(f"./benchmark_{args.language}.json", "w"), indent=4)

            
        # Remove model from memory
        del asr_pipeline
        import gc         
        gc.collect()
        torch.cuda.empty_cache() 
