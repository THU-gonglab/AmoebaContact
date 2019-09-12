# AmoebaContact
AmoebaContact is a program for multi-cutoff protein contact prediction which starts from target sequence alone. Different from traditional human-handed architecture picking, we used evolution algorithm to search neural networks more suitable for this particular field in AmoebaContact.
More details can be found in http://structpred.life.tsinghua.edu.cn/amoebacontact.html.

Tips for usage:
1. Package needs to be installed: BioPython & prody;
2. Blast package is also needed, install-command recommended by us is "sudo apt-get install ncbi-blast+";
3. Make sure you have got the tensorflow package of the newest version;
4. You should config your own addresses of HHLIB, CCMPRED, DEEPCNF, SPIDER3 and UNIPROT20. Code line 28-32, blanks filled with 'XX/XX' should be overrided.
5. The comments in the code marked all tips above and the different segmentations of the whole process.

