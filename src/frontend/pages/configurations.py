import os
import sys

import gradio as gr

# from app.core.configurations import get_configs

print(os.path.dirname(__file__))
print(os.path.curdir)

from core.configurations import get_configs

# from funcs.config_reading import read_config_file

x = get_configs()

def configurations():
	with gr.Blocks(title="Configurations"):
		gr.Markdown("## Configurations Files Parameters")
		gr.Markdown("Only edit if you know what you're doing.")

		# values = get_configs()
		# x = read_config_file()

		with gr.Column():
			with gr.Row():
				gr.Textbox(lines=1, value="Oi", placeholder="tchau")
				gr.Textbox(lines=1, value="Oi", placeholder="tchau")
				gr.Textbox(lines=1, value="Oi", placeholder="tchau")
			with gr.Row():
				gr.Textbox(lines=1, placeholder="tchau")
				gr.Textbox(lines=1, value="Oi", placeholder="tchau")
				gr.Textbox(lines=1, value="Oi", placeholder="tchau")