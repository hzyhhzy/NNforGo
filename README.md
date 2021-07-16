# NNforGo
Experiments on NN structures for Go/Gomoku   
----------------------------------------------
gat         GAT tests    
resnet      Resnet tests    
20k.npz     A small dataset for benchmark   

----------------------------------------------
Unzip Katago training data to data_p0/ , then run dataset_p0_to_p1.py. It will convert Katago data to a simpler format.   

Then run gat/auto**.sh to start train-shuffle loop   

benchmark.py is for speed testing. Just run "python benchmark.py --size xxxxxx". Detailed definition of --size can be seen in model.py   

----------------------------------------------
shuffle_keep.py is based on Katago's shuffle.py
