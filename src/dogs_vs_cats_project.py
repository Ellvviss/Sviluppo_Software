import gradio as gr
try:
    from ai import inference
except ImportError:
    from src.ai import inference

# Interfaccia Gradio
gr.Interface(fn=inference, inputs=gr.Image(image_mode="RGB", type="pil"), outputs="text").launch()
