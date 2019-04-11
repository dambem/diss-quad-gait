# for force in 20 30 40 50 60 70 80 90 100 110 120
# do
#   python laikago.py $force 0.008 0 ex2_1 n 10 6
# done
#
# for force in 20 30 40 50 60 70 80 90 100 110 120
# do
#   python laikago.py $force 0.008 1 ex2_2 n 10 6
# done
#
# for force in 20 30 40 50 60 70 80 90 100 110 120
# do
#   python laikago.py $force 0.008 2 ex2_3 n 10 6
# done


for leg in 4 6 8 10 12 14 16 18 20
do
  for hip in 4 6 8 10 12 14 16 18 20
  do
    for gait in 0 1 2
    do
      python laikago.py 70 0.008 $gait ex_ang n $leg $hip
    done
  done
done
