from Bio import SeqIO


# Função para extrair introns, exons e a espécie de um arquivo GenBank
def extract_dna_features(genbank_file):
    dataset = []
    total_records = 682100  # Total de registros no arquivo
    current_record = 0  # Contador para os registros processados
    
    # Iterar sobre cada sequência no arquivo GenBank
    for record in SeqIO.parse(genbank_file, "genbank"):
        current_record += 1
        species = "Unknown"

        if 334729 <= current_record <= 350000:
            print("gerangotango")
            continue
        
        # Tenta recuperar a informação da espécie no campo "source"
        for feature in record.features:
            if feature.type == "source":
                species = feature.qualifiers.get("organism", ["Unknown"])[0]
        
        # Itera sobre os features de íntrons e exons
        for feature in record.features:
            if feature.type in ["intron", "exon"]:
                # Verifica se a feature faz referência a outra sequência
                if "location_operator" in feature.qualifiers:
                    print(f"Ignorando feature com referência externa: {feature}")
                    continue

                # Extrai a sequência e adiciona ao dataset
                sequence = str(feature.extract(record.seq))
                feature_type = feature.type
                dataset.append([sequence, feature_type, species])
        
        # Feedback de progresso
        if current_record % 1000 == 0:
            print(f"Processado {current_record}/{total_records} registros.")

    print(f"Total de registros processados: {current_record}/{total_records}")
    return dataset

# Arquivo GenBank a ser processado
genbank_file = "./database/all.gb"

# Extrair as informações e montar o dataset
dataset = extract_dna_features(genbank_file)

# Salvar o dataset em um arquivo CSV
import csv

with open("dna_features_dataset.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Sequence", "Feature Type", "Species"])
    writer.writerows(dataset)

print("Dataset criado com sucesso!")
