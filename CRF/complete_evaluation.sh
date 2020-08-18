# This evaluation will also compute the accuracy
paste ../dataset/NL2SparQL4NLU.test.conll.txt result.txt | cut -f 1,2,4 > merge.txt
perl ../conlleval.pl -d "\t" < merge.txt > evaluation.txt
