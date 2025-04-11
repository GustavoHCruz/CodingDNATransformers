import random
import time

import matplotlib as plt


def train(dataset, modelo, epocas, taxa_aprendizado):
    # Simulando barra de progresso e resultados
    progresso = []
    acuracias = []
    for epoca in range(epocas):
        time.sleep(0.3)  # simula tempo de treino
        acc = 0.5 + random.random() * 0.5  # simula acurácia
        acuracias.append(acc)
        progresso.append((epoca + 1) / epocas)
    
    # Gera o gráfico
    plt.figure(figsize=(6, 3))
    plt.plot(range(1, epocas + 1), acuracias, marker='o')
    plt.title("Acurácia por Época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")
    
    return "Treinamento concluído com sucesso!", "accuracy_plot.png"