# Brain-Graph-SVM
Calculate graph measures from brain connectivity matrices and use them to classify cognitive state

This project calculates graph theoretical measures on brain connectivity matrices (128 x 128 regions) to classify which cognitive/clinical state a participant is in. These scripts were used for an anlysis aimed at classifying different levels of propofol anesthesia using a support vector machine, however they can easily be adapted to other experiments or other classification methods.

These scripts take brain connectivity matrices as input (these can be generated from FSL, SPM or other fMRI analysis software) and outputs a csv file containing the subject number, condition number and 5 graph measures (global efficiency, local efficiency, modularity, connector hubs and provincial hubs). 

Relevant publications:
Guimera R, Amaral L. Nature (2005) 433:895-900.
Bullmore E, Sporns O. Nature Reviews Neuroscience (2009) 10(3):186-98.
Cohen J, D'Esposito M. Journal of Neuroscience (2016) 36(48):12083-12094.
