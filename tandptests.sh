
echo "Beginning"
for leg in 10
do
    for oscillator in 0.010 0.008 0.006 0.004 0.002
    do
        # for gait in 0
        # do
        echo l $leg h $hip o $oscillator f $force
        python laikago.py $force $oscillator 0 big $leg $hip
        # done
  done
done
