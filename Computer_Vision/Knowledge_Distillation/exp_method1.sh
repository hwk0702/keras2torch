# python main.py --model teacher --epochs 100
# python main.py --model student --epochs 100

alpha_lst="0.1 0.25 0.5 0.75 0.9"
temperature_lst="1. 2.5 5. 7.5"
loss_method="method1"

for temp in $temperature_lst
do
    for alpha in $alpha_lst
    do
        for loss in $loss_method
        do
        echo '[KD] alpha: ' $alpha ', temperature: ' $temp
        python main.py --KD --model student --alpha $alpha --temperature $temp --epochs 100 --loss_method $loss
        done
    done
done

