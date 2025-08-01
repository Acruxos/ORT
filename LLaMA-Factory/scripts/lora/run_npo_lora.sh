#!/bin/bash
names=('1_Stephen_King' '2_Confucius' '3_Bruce_Lee' '4_Warren_Buffett' '5_Christina_Aguilera'
'6_Cindy_Crawford' '7_Marie_Osmond' '8_Paris_Hilton' '9_Justin_Bieber' '10_Prince_Harry,_Duke_of_Sussex'
'11_Miley_Cyrus' '12_Genghis_Khan' '13_Liza_Minnelli' '14_Taylor_Swift' '15_Mark_Cuban'
'16_Rhea_Perlman' '17_Mark_Hamill' '18_John_D._Rockefeller' '19_Alanis_Morissette' '20_Marlon_Brando'
'21_50_Cent' '22_Jim_Morrison' '23_Evel_Knievel' '24_Beyoncé' '25_Reba_McEntire'
'26_Justin_Timberlake' '27_Vanna_White' '28_Lil_Wayne' '29_Anna_Nicole_Smith' '30_Henry_Winkler'
'31_Leonardo_da_Vinci' '32_Kanye_West' '33_Paul_Walker' '34_Daniel_Day-Lewis' '35_Jim_Parsons'
'36_Henry_Kissinger' '37_Chuck_Norris' '38_Steven_Seagal' '39_Linda_Hamilton' '40_Danny_Trejo'
'41_Sam_Elliott' '42_Michael_Strahan' '43_Paul_Simon' '44_Meghan,_Duchess_of_Sussex' '45_Bruce_Springsteen'
'46_Raquel_Welch' '47_Lenny_Kravitz' '48_Bob_Saget' '49_Jon_Voight' '50_Ryan_Seacrest'
'51_Betty_White' '52_Chris_Brown' '53_Travis_Kelce' '54_Jay-Z' '55_Jackie_Chan'
'56_Mark_Harmon' '57_Whitney_Houston' '58_Rihanna' '59_Anderson_Cooper' '60_Brendan_Fraser'
'61_Tim_Burton' '62_Serena_Williams' '63_Dionne_Warwick' '64_Michelle_Pfeiffer' '65_Selena_Gomez'
'66_Kris_Jenner' '67_Hugh_Laurie' '68_Tom_Clancy' '69_John_Candy' '70_Vin_Diesel'
'71_Dakota_Fanning' '72_R._Kelly' '73_Emilio_Estevez' '74_Socrates' '75_Brooke_Shields'
'76_Bob_Barker' '77_Val_Kilmer' '78_Jennifer_Lopez' '79_Pamela_Anderson' '80_Tony_Blair'
'81_Vincent_van_Gogh' '82_Lindsay_Lohan' '83_Rebel_Wilson' '84_Nicolas_Cage' '85_Ted_Danson'
'86_John_Travolta' '87_Robert_Downey_Jr.' '88_Jason_Bateman' '89_Samuel_L._Jackson' '90_Karl_Marx'
'91_Halle_Berry' '92_Larry_Bird' '93_Johnny_Cash' '94_Chevy_Chase' '95_Bill_Paxton'
'96_Ice_Cube' '97_Don_Johnson' '98_Dwayne_Johnson' '99_RuPaul' '100_Matthew_Perry')

for name in "${names[@]}"
do
    mkdir -p ./logs/lora/npo_lora
    id=$name
    echo $id

    PYTHONPATH=./ WANDB_DISABLED=true python src/train_bash.py --stage npo --dpo_beta 0.2 \
    --model_name_or_path /data/yexiaotian/models/meta-llama/Meta-Llama-3-8B-Instruct --do_train --save_model \
    --dataset ${id}_Positive --dataset_dir ./data --finetuning_type lora --lora_target q_proj,v_proj \
    --output_dir ./saves/lora/llama3_8b_instruct/npo_lora --overwrite_cache \
    --overwrite_output_dir --cutoff_len 512 --preprocessing_num_workers 16 \
    --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine --logging_steps 10 --warmup_steps 20 --save_steps 30000 \
    --eval_steps 30000 --evaluation_strategy steps --load_best_model_at_end --template llama3 \
    --learning_rate 5e-5 --num_train_epochs 3.0 --val_size 0.0000001 --plot_loss \
    --output_result_dir ./results/lora/llama3_8b_instruct/npo_lora \
    --fp16 --eval_dataset_dir ./data/unlearn_bench/People/ \
    --target ${id} 2>&1 | tee ./logs/lora/npo_lora/${id}.log

    # 清空 checkpoint 文件夹
    find "${output_dir}" -mindepth 1 ! -name "train_results.json" ! -name "training_loss.png" ! -name "trainer_log.jsonl" -delete

done
