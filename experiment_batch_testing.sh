
echo "Beginning"
for leg in 10
do
  for hip in 10
  do
    for oscillator in  0.010 0.008 0.006 0.004 0.002  0.0018 0.0016 0.0014 0.0012 0.001 0.0008 0.0006 0.0004 0.0002
    do
      for force in 010 020 030 040 050 060 070 080 090 100 120
      do
        for gaits in 0 1 2
        do
        # for gait in 0
        # do
          echo l $leg h $hip o $oscillator f $force g $gaits
          python laikago_structural.py $force $oscillator $gaits struc $leg $hip
        # doned
        done
      done
    done
  done
done
