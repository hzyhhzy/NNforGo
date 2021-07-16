while true
do
	  python train.py --bs 120 --size 10 3 32 16 1 1 0 1 --save 10_3_32_16_ttft --gpu 0 &
	  python train.py --bs 120 --size 10 3 32 16 1 1 0 0 --save 10_3_32_16_ttff --gpu 1 &
	  sleep 2m
    cd ..
    python shuffle_keep.py -keep 2000000 &
    cd gat
	  wait
done

