
echo "Beginning experiments on seperate gaits"
for leg in 10
do
  for hip in 10
  do
    for oscillator in  0.010 0.008 0.006 0.004 0.002
    do
      for force in 020 030 040 050 060 070 080 090 100
      do
        for gaits in 0 1 2
        do
        # for gait in 0
        # do
          echo l $leg h $hip o $oscillator f $force g $gaits
          python laikago.py $force $oscillator $gaits gait_var $leg $hip
        # doned
        done
      done
    done
  done
done

echo "Beginning main experiment bulk on walking gaits"
for leg in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
  for hip in 10
  do
    for oscillator in  0.010 0.008 0.006 0.004 0.002
    do
      for force in 020 030 040 050 060 070 080 090 100
      do
        for gaits in 0
        do
        # for gait in 0
        # do
          echo l $leg h $hip o $oscillator f $force g $gaits
          python laikago.py $force $oscillator $gaits gait_var $leg $hip
        # doned
        done
      done
    done
  done
done

#
