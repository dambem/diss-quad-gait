echo "Beginning T-Tests, located in ttest folder"
for force in 20 30 40 50 60 70 80 90 100
do
  for oscillator in 0.010 0.008 0.006 0.004 0.002
  do
    # for gait in 0
    # do
    echo l $leg h $hip o $oscillator f $force
    python ttest.py $force $oscillator 0 ttest 5 5
    # done
  done
done
