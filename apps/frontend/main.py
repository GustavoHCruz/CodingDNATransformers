import gradio as gr
from pages.configurations import configurations
from pages.datasets import datasets
from pages.training import training

with gr.Blocks(title="Splicing Sites Identification") as app:
  gr.Markdown("# 🧬 Interface de Análise de Splicing com Transformers")

  with gr.Tabs():
    with gr.TabItem("📂 Datasets"):
      datasets()
    
    # with gr.TabItem("🧪 Teste"):
    #  aba_teste()
    
    with gr.TabItem("📈 Training"):
      training()

    with gr.TabItem("⚙️ Configurations"):
      configurations()

app.launch(show_api=False, share=False)
