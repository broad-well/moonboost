# Preprocessing

```
$ python .\data_preprocess.py --dataset_name vgm --dataset_folder ..\midi\starwars --output ..\datasetstarwars --model_config .\src\llama_recipes\configs\model_config.json --train_test_split_file None --train_ratio 0.9 --ts_threshold None
```

```
python data_preprocess.py  --dataset_name smalldata   --dataset_folder ..\midi_small   --output_folder ..\datasetsmall   --model_config src/llama_recipes/configs/model_config.json   --train_test_split_file None   --train_ratio 0.9   --ts_threshold None
```

# Installing torch with Cuda
```
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

# Training?? on dataset

```
$ python recipes\finetuning\real_finetuning_uncon_gen.py --lr 3e-4 --val_batch_size 2 --run_validation True  --validation_interval 10   --save_metrics True  --dist_checkpoint_root_folder checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen   --dist_checkpoint_folder ddp  --trained_checkpoint_path moonbeam_309M.pt --pure_bf16 True --enable_ddp False --use_peft True --peft_method lora --quantization False  --model_name ATEPP_bach  --dataset lakhmidi_dataset  --output_dir checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen  --batch_size_training 2 --context_length 2048  --num_epochs 300 --use_wandb False   --gamma 0.99
```

# Training?? on STARWARS and in datasets.oy @dataclass

### change the last 2 lines in this to be the correct paths
```
class lakhmidi_dataset:
    dataset: str = "lakhmidi_dataset"
    train_split: str = "train"
    test_split: str = "test"
    data_dir: str = "..\\datasetstarwars"
    csv_file: str = "..\\datasetstarwars\\train_test_split.csv"
```


&
```
$ python recipes\finetuning\real_finetuning_uncon_gen.py --lr 3e-4 --val_batch_size 2 --run_validation True  --validation_interval 10   --save_metrics True  --dist_checkpoint_root_folder checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen   --dist_checkpoint_folder ddp  --trained_checkpoint_path moonbeam_309M.pt --pure_bf16 True --enable_ddp False --use_peft True --peft_method lora --quantization False  --model_name ATEPP_bach  --dataset lakhmidi_dataset  --output_dir checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen  --batch_size_training 2 --context_length 2048  --num_epochs 300 --use_wandb False   --gamma 0.99
```
# Unconditional Running SW
```
python recipes/inference/custom_music_generation/unconditional_music_generation.py --csv_file ..\datasetstarwars\train_test_split.csv --top_p 0.95 --temperature 0.9  --model_config_path src/llama_recipes/configs/model_config.json   --ckpt_dir moonbeam_309M.pt  --finetuned_PEFT_weight_path .\checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen_datasetstarwars\9-0.safetensors  --tokenizer_path tokenizer.model  --max_seq_len 128   --max_gen_len 128  --max_batch_size 6   --num_test_data 2  --prompt_len 50

```
# Unconditional Running
```
python recipes/inference/custom_music_generation/unconditional_music_generation.py --csv_file ..\dataset\train_test_split.csv --top_p 0.95 --temperature 0.9  --model_config_path src/llama_recipes/configs/model_config.json   --ckpt_dir  moonbeam_309M.pt --finetuned_PEFT_weight_path .\checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen  --tokenizer_path tokenizer.model  --max_seq_len 512   --max_gen_len 512  --max_batch_size 6   --num_test_data 20  --prompt_len 50
```

```
python --nproc_per_node 1 recipes/inference/custom_music_generation/unconditional_music_generation.py   --csv_file ..\datasetstarwars\train_test_split.csv   --top_p 0.95  --temperature 0.9   --model_config_path src/llama_recipes/configs/model_config.json  --ckpt_dir moonbeam_309M.pt   --finetuned_PEFT_weight_path .\checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen_datasetstarwars\10-0.safetensors   --tokenizer_path tokenizer.model   --max_seq_len 512   --max_gen_len 512   --max_batch_size 6   --num_test_data 20   --prompt_len 50
```



#Training small midi
```
python recipes\finetuning\real_finetuning_uncon_gen.py --lr 3e-4 --val_batch_size 2 --run_validation True  --validation_interval 10   --save_metrics True  --dist_checkpoint_root_folder checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen_small   --dist_checkpoint_folder ddp  --trained_checkpoint_path moonbeam_309M.pt --pure_bf16 True --enable_ddp False --use_peft True --peft_method lora --quantization False  --model_name ATEPP_bach_small  --dataset lakhmidi_dataset  --output_dir checkpoints\finetuned_checkpoints\ATEPP_bach_uncon_gen_small  --batch_size_training 2 --context_length 2048  --num_epochs 300 --use_wandb False   --gamma 0.99
```