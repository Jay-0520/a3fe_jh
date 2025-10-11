from modeller import *
from modeller.automodel import loopmodel, assess
from modeller.automodel import refine

# === DEFINE GAPS ===
# First gap (residues 7-8)
LOOP1_A_START = '7'    # residue 7 on chain A
LOOP1_A_END   = '8'    # residue 8 on chain A
LOOP2_A_START = '115'  # residue 115 on chain A  
LOOP2_A_END   = '116'  # residue 116 on chain A 

# Second gap (residues 115-116)
LOOP1_B_START = str(174 + 7)   # = '181' (residue 7 on chain B)
LOOP1_B_END   = str(174 + 8)   # = '182' (residue 8 on chain B)
LOOP2_B_START = str(174 + 115) # = '289' (residue 115 on chain B)
LOOP2_B_END   = str(174 + 116) # = '290' (residue 116 on chain B)
# ===================

log.verbose()
env = environ()
env.io.atom_files_directory = ['.']

class DimerMultiLoopModel(loopmodel):
    def select_loop_atoms(self):
        sel = selection()
        # Select using MODELLER's sequential numbering
        sel.add(self.residue_range(LOOP1_A_START+':A', LOOP1_A_END+':A'))
        sel.add(self.residue_range(LOOP1_B_START+':B', LOOP1_B_END+':B'))
        sel.add(self.residue_range(LOOP2_A_START+':A', LOOP2_A_END+':A'))
        sel.add(self.residue_range(LOOP2_B_START+':B', LOOP2_B_END+':B'))
        
        return sel

a = DimerMultiLoopModel(env,
                        alnfile='dimer.ali',
                        knowns='dimer',      
                        sequence='target',   # You'll need to create this
                        loop_assess_methods=(assess.DOPE,))

a.starting_model = 1
a.ending_model   = 1         
a.loop.starting_model = 1
a.loop.ending_model   = 10   
a.loop.md_level = refine.fast

a.make()