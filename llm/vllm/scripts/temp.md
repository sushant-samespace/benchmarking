docker run -dit --gpus 1 --name whisper-better --network host -v /root/arun/whisper-better/:/app ai-asr

python3 -m pip install --no-cache-dir torch==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

python3 bench.py --filename ../audio/en_16.wav --language en --batch_sizes 2,4 --better

<!-- xformers 0.0.23 requires torch==2.1.1, but you have torch 2.1.0+cu121 which is incompatible. -->


python3 -m vllm.entrypoints.api_server --model ./models/Mistral-7B-v0.1-AWQ/ --quantization awq --swap-space 16 --disable-log-requests --dtype float16 --tensor-parallel-size 1

python3 -m vllm.entrypoints.api_server --model ./models/Mistral-7B-v0.1-AWQ/ --quantization awq --disable-log-requests --dtype float16 --tensor-parallel-size 1 --max-model-len 8192

python3 ./benchmark_serve.py --backend vllm --tokenizer ./models/Mistral-7B-v0.1-AWQ --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --batch-size 3

python3 ./benchmark_serve.py --backend vllm --tokenizer ./models/Mistral-7B-v0.1-AWQ --dataset ./new_data.json --batch-size 3

Sample O/P

{'text': ["Write a script that will create a sql server database table with schema mapping to the following Entity Framework class. Call the table ContactTypeAssignments.\n public virtual int Id { get; set; }\n\n public virtual int ContactTypeId { get; set; }\n\n public virtual int OfficeId { get; set; }\n\n public virtual int UserId { get; set; }\n\n public virtual CustomerContactType CustomerContactType { get; set; }\n\n public virtual Office Office { get; set; }\n\n public virtual User User { get; set; }\n\n public virtual DateTime CreatedOn { get; set; }\n\n public virtual DateTime ModifiedOn { get; set; }offsale;\n The first time this script is run it will drop the database and tables, and then re-create them, emptying the tables. Subsequent running of the script will recreate the tables, with data, but will not drop the database and existing tables.\n\n Windows project files (.sln,.csproj)\n   All deployment commands except cleanup use only the project file from the solution. The solution file is useful for including multiple database projects.\n\n   Any project file (.sln,.csproj)\n   Any command that generates physical output, defaults to the project file that is listed first in the `ProjectName` attribute in the `` section. This means that if a solution header is used where the only database project, then the solution file won't be included in the generated physical output. Conversely, if multiple database projects are listed in the ``, then the solution file will be used. Also, if multiple database projects are used with implicitly named `` or database projects are used with `ProjectReference` instead of `ProjectName` in the `` then it is possible to include a solution file and use the `OutputDir` parameter to define a directory path starting at the reaasured solution directory.\n   Commands in the DLL Deployment, Database libraries, and Database visualization categories that have a executable script as an input parameter and don't modify the project file, use the project file"]}

python3 ./benchmark_serve.py --backend vllm --tokenizer ./models/Mistral-7B-v0.1-AWQ --batch-size 3

python3 ./benchmark_serve.py --backend vllm --tokenizer ./models/Mistral-7B-v0.1-AWQ --batch-size 3 --use-beam-search