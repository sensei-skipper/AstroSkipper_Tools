#!/bash/bin

# generates sequencer with cluster size 60x3200, dynamic variable size 10x32, centers at (row,col) = (4,10) and (8,30) in cluster space, with heights and widths 2 (in cluster dims)
python write_sequencer.py --NROW 600 --NROWCLUSTER 60 --NCOL 3200 --NCOLCLUSTER 100 --NSAMP 100 --center 4,10 8,30 --height 2 --width 2
# similar sequencer to above, only changing centers and heights
python write_sequencer.py --NROW 600 --NROWCLUSTER 60 --NCOL 3200 --NCOLCLUSTER 100 --NSAMP 100 --center 1,10 9,30 --height 3 --width 3
# a sequencer with two rectangular regions (like we'd use in SIFS)
python write_sequencer.py --NROW 600 --NROWCLUSTER 600 --NCOL 3200 --NCOLCLUSTER 100 --NSAMP 100 --center 0,10 0,25 --height 1 --width 2
