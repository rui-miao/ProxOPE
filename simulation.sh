epiLenA=(1 2 4 8 16 24 32 48 64)
sampSizeA=(512)

epiLenB=(1 3 5)
sampSizeB=(256 512 1024 2048 4096)

for T in "${epiLenA[@]}"; do
    for n in "${sampSizeA[@]}"; do
        python ContSimuOffPolicy.py $T $n 0.2 cuda:0 50000 ResultA.csv 
    done
done

for T in "${epiLenB[@]}"; do
    for n in "${sampSizeB[@]}"; do
        python ContSimuOffPolicy.py $T $n 0.2 cuda:0 50000 ResultB.csv 
    done
done
