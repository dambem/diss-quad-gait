
echo "Beginning"
for force in 70 80 90 100 110 120 130
do
  for angle in 6 8 10 12 14 16 18 20 22 24 26 28 30
  do
    for gait in 0 1 2
    do
    # for gait in 0
    # do
    # echo l $leg h $hip o $oscillator f $force
    python ttest.py $force 0.008 $gait othergaitslarge 10 $angle
    # done
  done
  done
done
