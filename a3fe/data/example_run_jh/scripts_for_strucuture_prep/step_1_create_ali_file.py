# env = environ()
# aln = alignment(env)
# mdl = model(env, file='dimer.pdb')
# aln.append_model(mdl, align_codes='dimer', atom_files='dimer.pdb')
# aln.write(file='dimer2.ali', alignment_format='PIR')

from modeller import *

env = environ()
aln = alignment(env)

# Read the template structure
mdl = model(env, file='dimer.pdb')
aln.append_model(mdl, align_codes='dimer', atom_files='dimer.pdb')

# Extract sequence from the model and add target sequence
# Method 1: Try get_sequence()
try:
    template_seq = aln[0].get_sequence()
except AttributeError:
    # Method 2: Extract from residues
    template_seq = ''.join([res.code for res in aln[0].residues])

aln.append_sequence(template_seq)
aln[-1].code = 'target'

# Write the alignment
aln.write(file='dimer.ali', alignment_format='PIR')