while true
do
	  python train.py --size 20 6 16 8 1 1 0 0 --bs 384 --save ttff --gpu 0 &
	  python train.py --size 20 6 16 8 0 1 0 0 --bs 384 --save ftff --gpu 1 &
	  python train.py --size 20 6 16 8 1 0 0 0 --bs 384 --save tfff --gpu 2 &
	  python train.py --size 20 6 16 8 1 1 1 0 --bs 384 --save tttf --gpu 3 &
	  sleep 5m
    cd ..
    python shuffle_keep.py -keep 5000000 &
    cd gat
	  wait
done

