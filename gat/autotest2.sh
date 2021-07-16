while true
do
	  
	  python train.py --size 20 6 16 8  0 1 0 0  1 0 0 0 1 --bs 384 --save 3x3conv --gpu 0 --maxstep 10000 &
	  python train.py --size 20 6 16 8  0 1 0 0  1 1 0 0 0 --bs 384 --save AttActi --gpu 1 &
	  python train.py --size 20 6 16 8  0 1 0 0  1 0 0 1 0 --bs 384 --save EarlyVRelu --gpu 2 &
	  python train.py --size 20 6 16 8  0 0 0 0  1 0 0 1 0 --bs 384 --save EarlyVRelu2 --gpu 3 &
	  sleep 5m
    cd ..
    python shuffle_keep.py -keep 5000000 &
    cd gat
	  wait
done

