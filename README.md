
# Evaluating Machine Learning Approaches for Predicting Human Infection Risk from Coronavirus Spike Protein Sequences

Please see below the description of Supplemental Data, Basic Workflow, and how to re-create Transformer embeddings.

## SUPPLEMENTAL DATA

Files:
 * cv_bad_split_7_fold.csv - Supplemental data from simple splitting experiment
 * cv_group_7_fold.csv - Supplemental data from species-aware splitting experiment
 * output.txt - Stdout from experiments

## HOW TO RUN THE PROJECT

Files needed for basic project workflow:
 * Sequences.fasta - Data set from Kuzmin et al.
 * kuzmin_model_776.py - Code to generate project results
 * embeddings.pkl - Output embeddings created previously from make_embeddings.ipynb

Basic Workflow:
	1. Make sure that the working directory contains Sequences.fasta, the input data for the project. 
	2. Make sure the working directory contains embeddings.pk1, the pickled embedding representations of the sequences.
	3. Run the file kuzmin_model_776.py. No arguments are needed. Pipe to a text file to preserve stdout, if desired. Some fold metadata is printed to the screen as shown in the included output.txt.
	4. The project will run, generating cv_bad_split_7_fold.csv, cv_group_7_fold.csv, and the figures from the paper. 

## HOW TO RE-CREATE TRANSFORMER EMBEDDINGS

File needed:
 * Sequences.fasta - Data set from Kuzmin et al.
 * kuzmin_model_776.py - Commented code in Lines 173-188 will generate split_seqs.faa.
 * make_embeddings.ipynb - Code to generate Transformer embeddings from split sequences.
 
Intermediate file created:
 * split_seqs.faa - Sequence fragments needed for make_embeddings.ipynb

File produced for Basic Workflow:
 * embeddings.pkl - Output embeddings from make_embeddings.ipynb

STEP 1: Split the sequences.
	a. Uncomment lines 173-188 in kuzmin_model_776.py. Comment all lines after line 188.
	b. Make sure that the working directory contains Sequences.fasta, the input data, and run this edited version of kuzmin_model_776.py. It will save split_seqs.faa.

STEP 2: Run the split sequences through the Transformer.
	a. Make sure split_seqs.faa is in your working directory.
	b. Run make_embeddings.ipynb. It will save embeddings.pk1.

STEP 3: Follow the "Basic Workflow", as above.

## FILE DEFINITION SUMMARY

 * Sequences.fasta - Data set from Kuzmin et al.
 * kuzmin_model_776.py - Code to generate project results
 * make_embeddings.ipynb - Code to generate Transformer embeddings
 * split_seqs.faa - Sequence fragments needed for make_embeddings.ipynb
 * embeddings.pkl - Output embeddings from make_embeddings.ipynb
 * cv_bad_split_7_fold.csv - Supplemental data from simple splitting experiment
 * cv_group_7_fold.csv - Supplemental data from species-aware splitting experiment
 * output.txt - Stdout from experiments
