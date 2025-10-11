# protein_final.pdb is target.BL00020001.pdb 
pdb4amber -i protein_final.pdb -o protein_final_fixed.pdb --add-missing-atoms --keep-altlocs 

# gmx pdb2gmx -f protein_final.pdb -o protein_final_fixed.pdb
