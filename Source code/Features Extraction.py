#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from Bio.SeqUtils import molecular_weight, MeltingTemp
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from Bio.Seq import Seq

# Define a function to calculate features for a DNA sequence
def calculate_features(dna_sequence):
    # 1. Number of codons in the DNA sequence
    num_codons = len(dna_sequence) // 3

    # 2. Number of occurrences of each codon in the DNA sequence
    codon_counts = Counter([dna_sequence[i:i+3] for i in range(0, len(dna_sequence), 3)])

    # 3. Proportion of each codon in the DNA sequence
    codon_proportions = {codon: count / num_codons for codon, count in codon_counts.items()}

    # 4. Number of nucleotides in the sequence
    num_nucleotides = len(dna_sequence)

    # 5. Average number of nucleotides per codon in the DNA sequence
    avg_nucleotides_per_codon = num_nucleotides / num_codons

    # 6. Percentage of GC
    gc_percentage = (dna_sequence.count('G') + dna_sequence.count('C')) / num_nucleotides * 100

    # 7. Percentage of purines AG
    purines_percentage = (dna_sequence.count('A') + dna_sequence.count('G')) / num_nucleotides * 100

    # 8. Percentage of pyrimidines CT
    pyrimidines_percentage = (dna_sequence.count('C') + dna_sequence.count('T')) / num_nucleotides * 100

    # 9. Percentage of AT
    at_percentage = (dna_sequence.count('A') + dna_sequence.count('T')) / num_nucleotides * 100

    # 10. Molecular weight of the sequence
    mw = molecular_weight(dna_sequence)

    # 11. Melting temperature of the sequence
    melting_temp = MeltingTemp.Tm_GC(dna_sequence)

    # 12-15. Proportion of each nucleotide in the DNA sequence
    a_proportion = dna_sequence.count('A') / num_nucleotides
    c_proportion = dna_sequence.count('C') / num_nucleotides
    g_proportion = dna_sequence.count('G') / num_nucleotides
    t_proportion = dna_sequence.count('T') / num_nucleotides

    # Convert DNA to protein sequence and calculate protein features
    protein_sequence = str(Seq(dna_sequence).transcribe().translate())

    # Remove any asterisks '*' from the protein sequence
    protein_sequence = protein_sequence.replace("*", "")
    
    # 17. Number of amino acids
    num_amino_acids = len(protein_sequence)

    # 18. Percentage of amino acids
    percentage_amino_acids = {aa: count / num_amino_acids * 100 for aa, count in ProteinAnalysis(protein_sequence).get_amino_acids_percent().items()}

    # 19. Aromaticity
    aromaticity = ProteinAnalysis(protein_sequence).aromaticity()

    # 20. Instability index
    instability_index = ProteinAnalysis(protein_sequence).instability_index()

    # 21. Isoelectric point
    isoelectric_point = ProteinAnalysis(protein_sequence).isoelectric_point()

    # 22. Molecular weight
    protein_mw = ProteinAnalysis(protein_sequence).molecular_weight()

    # 23. Gravy (Grand Average Hydropathy)
    gravy = ProteinAnalysis(protein_sequence).gravy()

    # Return all the calculated features as a dictionary
    return {
        "NumCodons": num_codons,
        **dict(codon_counts),  # Include CodonCounts as separate columns
        "NumNucleotides": num_nucleotides,
        "AvgNucleotidesPerCodon": avg_nucleotides_per_codon,
        "GCPercentage": gc_percentage,
        "PurinesPercentage": purines_percentage,
        "PyrimidinesPercentage": pyrimidines_percentage,
        "ATPercentage": at_percentage,
        "MolecularWeight": mw,
        "MeltingTemperature": melting_temp,
        "AProportion": a_proportion,
        "CProportion": c_proportion,
        "GProportion": g_proportion,
        "TProportion": t_proportion,
        "NumAminoAcids": num_amino_acids,
        **dict(percentage_amino_acids),
        "Aromaticity": aromaticity,
        "InstabilityIndex": instability_index,
        "IsoelectricPoint": isoelectric_point,
        "ProteinMolecularWeight": protein_mw,
        "Gravy": gravy,
    }

# Read the CSV file containing DNA sequences and labels
#df = pd.read_csv("E:/R(4) Sajeeb Sir/Datasets/A.thaliana.csv")
#df = pd.read_csv("E:/R(4) Sajeeb Sir/Datasets/C_elegens.csv")
#df = pd.read_csv("E:/R(4) Sajeeb Sir/Datasets/D.melanogaster.csv")
#df = pd.read_csv("E:/R(4) Sajeeb Sir/Datasets/E.coli.csv")
#df = pd.read_csv("E:/R(4) Sajeeb Sir/Datasets/F_vesca.csv")
#df = pd.read_csv("E:/R(4) Sajeeb Sir/Datasets/G.pickeringi.csv")
#df = pd.read_csv("E:/R(4) Sajeeb Sir/Datasets/G.subterraneus.csv")
df = pd.read_csv("E:/R(4) Sajeeb Sir/Datasets/R_chinensis.csv")

# Calculate features for each DNA sequence in the "sequences" column
feature_list = []
for index, row in df.iterrows():
    dna_sequence = row["Sequences"]  # Replace "sequences" with your column name
    features = calculate_features(dna_sequence)
    feature_list.append(features)

# Create a DataFrame from the list of features, including the "label" column
feature_df = pd.DataFrame(feature_list)
feature_df["label"] = df["label"]  # Add the "label" column from your dataset

feature_df.isnull().any()
feature_df[feature_df.isnull().any(axis=1)]
feature_df.fillna(feature_df.mean(), inplace=True)
feature_df[feature_df.isnull().any(axis=1)]
df1 = feature_df.drop_duplicates(keep='first')

# Exclude the "label" column before scaling
columns_to_scale = df1.columns.difference(["label"])

# Apply Min-Max scaling to the selected columns
scaler = MinMaxScaler()
df1[columns_to_scale] = scaler.fit_transform(df1[columns_to_scale])

# Save the DataFrame to a new CSV file with the calculated features
#df1.to_csv("E:/R(4) Sajeeb Sir/Datasets/A.thaliana(PreProcessed).csv")
#df1.to_csv("E:/R(4) Sajeeb Sir/Datasets/C_elegens(PreProcessed).csv")
#df1.to_csv("E:/R(4) Sajeeb Sir/Datasets/D.melanogaster(PreProcessed).csv")
#df1.to_csv("E:/R(4) Sajeeb Sir/Datasets/E.coli(PreProcessed).csv")
#df1.to_csv("E:/R(4) Sajeeb Sir/Datasets/F_vesca(PreProcessed).csv")
#df1.to_csv("E:/R(4) Sajeeb Sir/Datasets/G.pickeringi(PreProcessed).csv")
#df1.to_csv("E:/R(4) Sajeeb Sir/Datasets/G.subterraneus(PreProcessed).csv")
df1.to_csv("E:/R(4) Sajeeb Sir/Datasets/R_chinensis(PreProcessed).csv")
