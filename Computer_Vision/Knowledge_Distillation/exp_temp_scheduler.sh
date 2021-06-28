# python main.py --model teacher --epochs 100
# python main.py --model student --epochs 100

alpha_lst=("0.9" "0.5")
temperature_lst=("5." "10")

for i in {0..1}
do    
echo '[KD] alpha: ' ${alpha_lst[i]} ', temperature: ' ${temperature_lst[i]}
python main.py --KD --model student --alpha ${alpha_lst[i]} --temperature ${temperature_lst[i]} \
               --epochs 100 --loss_method method1 --temp_scheduler

done

echo '[KD] alpha: 0.5, temperature: 7.5'
python main.py --KD --model student --alpha 0.5 --temperature 7.5 \
               --epochs 100 --loss_method method2 --temp_scheduler
            

