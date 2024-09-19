import pandas as pd
# 创建密码子到数字的映射字典
codon_to_number = {
    'ATA': 1, 'ATC': 2, 'ATT': 3, 'ATG': 4,
    'ACA': 5, 'ACC': 6, 'ACG': 7, 'ACT': 8,
    'AAC': 9, 'AAT': 10, 'AAA': 11, 'AAG': 12,
    'AGA': 13, 'AGC': 14, 'AGG': 15, 'AGT': 16,
    'CTA': 17, 'CTC': 18, 'CTG': 19, 'CTT': 20,
    'CCA': 21, 'CCC': 22, 'CCG': 23, 'CCT': 24,
    'CAC': 25, 'CAT': 26, 'CAA': 27, 'CAG': 28,
    'CGA': 29, 'CGC': 30, 'CGG': 31, 'CGT': 32,
    'GTA': 33, 'GTC': 34, 'GTG': 35, 'GTT': 36,
    'GCA': 37, 'GCC': 38, 'GCG': 39, 'GCT': 40,
    'GAC': 41, 'GAT': 42, 'GAA': 43, 'GAG': 44,
    'GGA': 45, 'GGC': 46, 'GGG': 47, 'GGT': 48,
    'TCA': 49, 'TCC': 50, 'TCG': 51, 'TCT': 52,
    'TTC': 53, 'TTT': 54, 'TTA': 55, 'TTG': 56,
    'TAC': 57, 'TAT': 58, 'TAA': 59, 'TAG': 60,
    'TGC': 61, 'TGT': 62, 'TGA': 63, 'TGG': 64,
    'TGT': 65, 'TGC': 66, 'TGG': 67, 'TGT': 68
}

# 创建氨基酸到数字的映射字典
AA_to_number = {
    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10, 'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20
}


def convert_cds_to_numbers(cds_sequence):
    cds_sequence = cds_sequence.upper()
    if len(cds_sequence) % 3 != 0:
        raise ValueError("CDS sequence length must be a multiple of 3.")
    codons = [cds_sequence[i:i + 3] for i in range(0, len(cds_sequence), 3)]
    codon_numbers = [codon_to_number.get(codon, None) for codon in codons]
    return codon_numbers

def convert_protein_seq_to_numbers(protein_sequence):
    protein_sequence = protein_sequence.upper()
    aa_numbers = [AA_to_number.get(aa, None) for aa in protein_sequence]
    return aa_numbers

def read_csv_to_numbers(file_path, column_name, seq):
    df = pd.read_csv(file_path)
    sequence_column = df[column_name]
    if seq == 'CDS':
        sequence_numbers = [convert_cds_to_numbers(sequence) for sequence in sequence_column]
    else:
        sequence_numbers = [convert_protein_seq_to_numbers(sequence) for sequence in sequence_column]
    return sequence_numbers



# test
cds_sequence = "ATGGCCATTGTAATGGGCCGCTGA"
codon_numbers = convert_cds_to_numbers(cds_sequence)
print(codon_numbers)

cds_sequence = "AMAMAM"
codon_numbers = convert_protein_seq_to_numbers(cds_sequence)
print(codon_numbers)