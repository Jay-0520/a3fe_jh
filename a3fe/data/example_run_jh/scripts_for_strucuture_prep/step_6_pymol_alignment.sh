# Simple version - save as align.pml
cat > align.pml << 'EOF'
delete all
load protein.pdb, target
load protein_final_fixed.pdb, mobile

# Exclude loop regions from alignment
select align_atoms, mobile and name CA and not (resi 7-8+115-116+181-182+289-290)
select target_atoms, target and name CA and not (resi 7-8+115-116+181-182+289-290)

# Perform alignment
align align_atoms, target_atoms

# Save result
save protein_final_fixed_aligned.pdb, mobile
quit
EOF

# Run PyMOL
pymol -c -q align.pml
