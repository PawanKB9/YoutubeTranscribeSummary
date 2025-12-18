import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time


# ---------------- Streamlit Config ----------------
st.set_page_config(layout="wide")


# ---------------- Helper Functions ----------------
def download_video(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    return audio_stream.download()


def initialize_model(model_path):
    return PromptModel(
        model_name_or_path=model_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )


def initialize_prompt_node(model):
    return PromptNode(
        model_name_or_path=model,
        default_prompt_template="deepset/summarization",
        use_gpu=False
    )


def transcribe_and_summarize(file_path, prompt_node):
    whisper = WhisperTranscriber()
    pipeline = Pipeline()

    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])

    result = pipeline.run(file_paths=[file_path])
    return result


# ---------------- Main App ----------------
def main():

    st.title("YouTube Video Summarizer")
    st.subheader("Simple YouTube video summary app using AI")

    with st.expander("About the App"):
        st.write(
            "This app converts YouTube video audio into text and generates a short summary. "
            "It is made for learning purposes and runs on CPU."
        )

    youtube_url = st.text_input("Enter YouTube URL")

    if st.button("Submit") and youtube_url:
        start_time = time.time()

        try:
            # Download audio
            file_path = download_video(youtube_url)

            # Load model
            model_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
            model = initialize_model(model_path)
            prompt_node = initialize_prompt_node(model)

            # Transcribe + summarize
            output = transcribe_and_summarize(file_path, prompt_node)

            summary = output["results"][0].split("\n\n[INST]")[0]

            end_time = time.time()
            elapsed_time = end_time - start_time

            col1, col2 = st.columns(2)

            with col1:
                st.video(youtube_url)

            with col2:
                st.header("Video Summary")
                st.success(summary)
                st.write(f"Time taken: {elapsed_time:.2f} seconds")

        except Exception:
            st.error("Something went wrong. Please check the YouTube link and try again.")


if __name__ == "__main__":
    main()
