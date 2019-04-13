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
for leg in 05
do
  for hip in 05
  do
    for oscillator in 0.010 0.008 0.006 0.004 0.002
    do
      for force in 020 030 040 050 060 070 080 090 100 120
      do
        # for gait in 0
        # do
        echo l $leg h $hip o $oscillator f $force
        python laikago.py $force $oscillator 0 big $leg $hip
        # done
      done
    done
  done
done
echo "Beginning"
for leg in 05
do
  for hip in 09
  do
    for oscillator in 0.006 0.004 0.002
    do
      for force in 020 030 040 050 060 070 080 090 100 120
      do
        # for gait in 0
        # do
        echo l $leg h $hip o $oscillator f $force
        python laikago.py $force $oscillator 0 big $leg $hip
        # done
      done
    done
  done
done

echo "Beginning"
for leg in 10 11 12 13 14 15 16 17  18 19 20
do
  for hip in 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
  do
    for oscillator in 0.010 0.008 0.006 0.004 0.002
    do
      for force in 020 030 040 050 060 070 080 090 100 120
      do
        # for gait in 0
        # do
        echo l $leg h $hip o $oscillator f $force
        python laikago.py $force $oscillator 0 big $leg $hip
        # done
      done
    done
  done
done
10 6
60
70
80
