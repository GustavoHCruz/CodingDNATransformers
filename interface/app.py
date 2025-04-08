import gradio as gr
from transformers import pipeline

modelos = {
	"Modelo 1 (Donor)": "seu_user/modelo_donor",
	"Modelo 2 (Acceptor)": "seu_user/modelo_acceptor",
	"Modelo 3 (Splicing completo)": "seu_user/modelo_completo"
}

datasets = {
	"Dataset Fungos": "data/fungos.fasta",
	"Dataset Humanos": "data/human.fasta"
}

def preditor(modelo, dataset, sequencia, max_length, threshold):
	pipe = pipeline("text-classification", model=modelos[modelo])
	result = pipe(sequencia, truncation=True, max_length=max_length)
	pred = result[0]
	if pred['score'] > threshold:
		return f"{pred['label']} ({pred['score']:.2f})"
	return "Sem predição confiável"

with gr.Blocks() as demo:
	gr.Markdown("# 🧬 Identificação de Splicing Sites")
	
	with gr.Row():
		modelo_sel = gr.Dropdown(choices=list(modelos.keys()), label="Modelo alvo")
		dataset_sel = gr.Dropdown(choices=list(datasets.keys()), label="Dataset alvo")
	
	sequencia_input = gr.Textbox(label="Sequência DNA", placeholder="Digite a sequência de nucleotídeos (A, T, G, C)")
	max_len = gr.Slider(minimum=32, maximum=512, value=128, step=32, label="Tamanho máximo da sequência")
	threshold = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, label="Limiar de confiança")

	btn = gr.Button("Classificar")
	output = gr.Textbox(label="Resultado")

	btn.click(fn=preditor, inputs=[modelo_sel, dataset_sel, sequencia_input, max_len, threshold], outputs=output)

demo.launch()