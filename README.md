# AmoebaContact
AmoebaContact is a program for multi-cutoff protein contact prediction which starts from target sequence alone. Different from traditional human-handed architecture picking, we used evolution algorithm to search neural networks more suitable for this particular field in AmoebaContact.
More details can be found in http://structpred.life.tsinghua.edu.cn/amoebacontact.html.

# Contents explination:
1. "run.py" is for architecture searching, the searching method is mainly coded in the package "autoML".
2. "finetune.py" is for finetuning model of different cutoffs.
3. "AmoebaContact.py" intergrates feature production, extraction and the whole inference process using our well-trained model, which is stored in the fold "models".

# Tips for usage:
## "run.py"
1. arguments of this scripts are :
        '--data_dir' (input data path)
	      '--model_dir' (output model path)
	      '--stack_num' (number of blocks stacked in the whole architecture)
	      '--combine_num' (number of operators in one block)
	      '--op_depth' (number of channel for every operator)
## "finetune.py"
1. This script would write a shell script called "run.sh". When finished, execute that script by "bash ./run.sh" for model finetuning. 
## "AmoebaContact.py"
1. Package needs to be installed: BioPython & prody;
2. Blast package is also needed, install-command recommended by us is "sudo apt-get install ncbi-blast+";
3. Make sure you have got the tensorflow package of the newest version;
4. You should config your own addresses of HHLIB, CCMPRED, DEEPCNF, SPIDER3 and UNIPROT20. Code line 28-32, blanks filled with 'XX/XX' should be overrided.
5. The comments in the code marked all tips above and the different segmentations of the whole process.

