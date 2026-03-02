import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr
from preprocess import preprocess_text
from predict import predict_review

def process_review(review):
    if not review or review.strip() == "":
        return "Please enter a review", "N/A", "N/A"
    
    try:
        preprocessed = preprocess_text(review)
        sentiment, intent, topic = predict_review(preprocessed)
        
        sentiment_output = f"**{sentiment.upper()}**"
        intent_output = f"**{intent.upper()}**"
        topic_output = f"**Topic {topic}**"
        
        return sentiment_output, intent_output, topic_output
    
    except Exception as e:
        return f"Error: {str(e)}", "Error", "Error"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📝 Sentiment Analysis & Intent Classification")
    gr.Markdown("Analyze customer reviews to extract sentiment, intent, and topic.")
    
    with gr.Row():
        with gr.Column():
            review_input = gr.Textbox(
                label="Review Text",
                placeholder="Enter a customer review here...",
                lines=5,
                interactive=True
            )
            submit_btn = gr.Button("Analyze Review", variant="primary", size="lg")
    
    gr.Markdown("### Prediction Results")
    
    with gr.Row():
        with gr.Column():
            sentiment_output = gr.Textbox(
                label="Sentiment",
                interactive=False,
                text_align="center"
            )
        with gr.Column():
            intent_output = gr.Textbox(
                label="Intent",
                interactive=False,
                text_align="center"
            )
        with gr.Column():
            topic_output = gr.Textbox(
                label="Topic",
                interactive=False,
                text_align="center"
            )
    
    submit_btn.click(
        fn=process_review,
        inputs=review_input,
        outputs=[sentiment_output, intent_output, topic_output]
    )
    
    review_input.submit(
        fn=process_review,
        inputs=review_input,
        outputs=[sentiment_output, intent_output, topic_output]
    )
    
    gr.Examples(
        examples=[
            ["The product arrived late and is completely broken. Very disappointed!"],
            ["Love this product! Best purchase ever made."],
            ["Can I return this? I need my money back."],
            ["Delivery was fast, but the item is damaged."]
        ],
        inputs=review_input,
        outputs=[sentiment_output, intent_output, topic_output],
        fn=process_review,
        cache_examples=False
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
