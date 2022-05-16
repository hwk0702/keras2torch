# python main.py --exp-name 'vit_base_patch16_224' --batch-size 32 
python main.py --exp-name 'vit_base_patch16_224+LoRA_r8_a8' --batch-size 32 --apply_lora --lora_r 8 --lora_alpha 8
python main.py --exp-name 'vit_base_patch16_224+LoRA_r4_a8' --batch-size 32 --apply_lora --lora_r 4 --lora_alpha 8
python main.py --exp-name 'vit_base_patch16_224+LoRA_r2_a8' --batch-size 32 --apply_lora --lora_r 2 --lora_alpha 8
