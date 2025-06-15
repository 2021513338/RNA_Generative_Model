import pandas as pd
from Sequence_Convert import load_data_from_csv
from RNN_CRF_Train import train_RNN_CRF
from Transformer_Train import train_Transformer
from GPT_Train import train_GPT
from BERT_Train import train_bert_mlm, train_regression
'''
file_path = "CCDS_Seq_test.csv"
protein_column = "Sequence"
cds_column = "CDS"

amino_acid_seqs, labels = load_data_from_csv(file_path, protein_column, cds_column)

#train_RNN_CRF(amino_acid_seqs, labels, 256, 100, 2)

train_Transformer(amino_acid_seqs, labels, 8, 6, 128, 1)
'''
file_path = "five_prime_utr_test.csv"
UTR_column = "Sequence"
df = pd.read_csv(file_path)
utr_seqs = df[UTR_column]

train_GPT(utr_seqs, 'UTR_RNA_vec.model', "best_GPT.pth", 10)

'''
file_path = "BERT_test.csv"
cds_column = "CDS_Sequence"
utr_5_colum = "UTR5_Sequence"
utr_3_colum = "UTR3_Sequence"
label_colum = "CAI"
df = pd.read_csv(file_path)
CDS_sequences = df[cds_column]
Five_UTR_sequences = df[utr_5_colum]
Three_UTR_sequences = df[utr_3_colum]
labels = df[label_colum]
#train_bert_mlm(CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, 3, "bert_RNA_vec.model", 1)
train_regression( CDS_sequences, Five_UTR_sequences, Three_UTR_sequences, labels, 3, "bert_RNA_vec.model", 1)
'''