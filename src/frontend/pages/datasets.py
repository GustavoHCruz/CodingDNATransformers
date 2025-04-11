import gradio as gr
from core.training import train


def datasets():
	with gr.Blocks(title="Datasets"):
		gr.Markdown("## Ainda não sei")
				
		with gr.Row():
			dataset_input = gr.File(label="Dataset CSV")
			modelo_dropdown = gr.Dropdown(["Transformer A", "Transformer B", "Meu Modelo"], label="Modelo")
		
		with gr.Row():
			epocas_input = gr.Slider(1, 20, value=5, label="Épocas")
			taxa_aprendizado_input = gr.Slider(0.0001, 0.1, value=0.001, label="Taxa de Aprendizado")
		
		botao_treinar = gr.Button("Iniciar Treinamento")
		saida_texto = gr.Textbox(label="Status")
		grafico_output = gr.Image(label="Gráfico de Acurácia")

		botao_treinar.click(
			fn=train,
			inputs=[dataset_input, modelo_dropdown, epocas_input, taxa_aprendizado_input],
			outputs=[saida_texto, grafico_output]
		)
