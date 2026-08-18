# DICEE

dicee's ReadFromDisk matches 'train'/'valid'/'test' as substrings of
the *full* glob path, so any of those words in an ancestor directory
silently misroutes every split file. Therefore the directory can only
contain these substrings if they also contain train.txt, valid.txt,
and test.txt