for force in 20, 30, 40, 50, 60, 70, 80
do
for leg in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
  for hip in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
  do
      python laikago.py 70 0.008 $gait ex_ang n $leg $hip
  done
done
done
