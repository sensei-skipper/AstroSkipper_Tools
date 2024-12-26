#!/bash/bin

# generates sequencer with cluster size 60x3200, dynamic variable size 10x32, centers at (row,col) = (4,10) and (8,30) in cluster space, with heights and widths 2 (in cluster dims)

# Q1017-207 (only Lyman-alpha)
python3 write_sequencer.py --NROW 600 --NROWCLUSTER 600 --NCOL 3400 --NCOLCLUSTER 20 --NSAMP 70 --center 0,50 0,165  --height 1 1 --width 29 5 --filename "sample_sequencer_HB89_1159_123_NSAMP_50_OS_COL_100.xml"
