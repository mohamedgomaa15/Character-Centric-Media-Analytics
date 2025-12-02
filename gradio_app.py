import gradio as gr
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, NetworkCharacterGenerator
from text_classifier import JutsuClassifier
from character_chatbot import CharacterChatbot
from dotenv import load_dotenv
import os

load_dotenv()

def get_themes(theme_list, subtitles_path, save_path):
    theme_list = theme_list.split(',')
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    theme_list = [theme for theme in theme_list if theme != "dialogue"]
    output_df = [theme_list].sum().reset_index()
    output_df.columns = ["theme", "score"]

    output_chars = gr.BarPlot(
        output_df,
        x="theme",
        y="score",
        title="Series Theme",
        tooltip=["Theme", "Score"],
        vertical=False,
        width=500,
        height=250
    )
    return output_chars

def get_character_network(sentence_path, save_path):
    named_entity_recognizer = NamedEntityRecognizer()
    df = named_entity_recognizer.get_ners(sentence_path, save_path)

    character_network = NetworkCharacterGenerator()
    relationship_df = character_network.generate_character(df)
    html_out = character_network.network_graph(relationship_df)
    return html_out

def classify_text(text_classifcation_model,text_classifcation_data_path,text_to_classify):
    jutsu_classifier = JutsuClassifier(model_path = text_classifcation_model,
                                       data_path = text_classifcation_data_path,
                                       huggingface_token = os.getenv('huggingface_token'))
    
    output = jutsu_classifier.classify_jutsu(text_to_classify)
    output = output[0]
    
    return output

def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatbot("MohamedGomaa/Naruto_Llama-3-8B",
                                         huggingface_token = os.getenv('huggingface_token')
                                         )

    output = character_chatbot.chat(message, history)
    output = output['content'].strip()
    return output

def main():

    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero Shot Classification)</h1>")
                with gr.Row():
                    with gr.Column():
                        plot = gr.BarPlot()
                    with gr.Column():
                        theme_list = gr.Textbox(label="Theme")
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="Save Path")
                        themes_button = gr.Button("Get Theme")
                        themes_button.click(get_themes, inputs=[theme_list, subtitles_path, save_path], outputs=[plot])


        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NER)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html = gr.HTML()
                    with gr.Column():
                        subtitles_path = gr.Textbox(label="Subtitles or Script Path")
                        save_path = gr.Textbox(label="NERs Save Path")
                        ner_button = gr.Button("Get Character Network")
                        ner_button.click(get_character_network, inputs=[subtitles_path, save_path], outputs=[network_html])

        # Text Classification with LLMs
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output = gr.Textbox(label="Text Classification Output")
                    with gr.Column():
                        text_classifcation_model = gr.Textbox(label='Model Path')
                        text_classifcation_data_path = gr.Textbox(label='Data Path')
                        text_to_classify = gr.Textbox(label='Text input')
                        classify_text_button = gr.Button("Clasify Text (Jutsu)")
                        classify_text_button.click(classify_text, inputs=[text_classifcation_model,text_classifcation_data_path,text_to_classify], outputs=[text_classification_output])

        # Character Chatbot Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Chatbot</h1>")
                gr.ChatInterface(chat_with_character_chatbot)



    iface.launch(share=True)

if __name__ == '__main__':
    main()