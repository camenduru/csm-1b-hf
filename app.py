import os

import gradio as gr
import numpy as np
import spaces
import torch
import torchaudio
from generator import Segment, load_csm_1b
from huggingface_hub import hf_hub_download, login
from watermarking import watermark

# api_key = os.getenv("HF_TOKEN")
gpu_timeout = int(os.getenv("GPU_TIMEOUT", 60))
# CSM_1B_HF_WATERMARK = list(map(int, os.getenv("WATERMARK_KEY").split(" ")))
CSM_1B_HF_WATERMARK = "CSM_1B_HF_WATERMARK"

# login(token=api_key)

SPACE_INTRO_TEXT = """\
# Sesame CSM 1B

Generate from CSM 1B (Conversational Speech Model). 
Code is available on GitHub: [SesameAILabs/csm](https://github.com/SesameAILabs/csm). 
Checkpoint is [hosted on HuggingFace](https://huggingface.co/sesame/csm-1b).

Try out our interactive demo [sesame.com/voicedemo](https://www.sesame.com/voicedemo), 
this uses a fine-tuned variant of CSM.

The model has some capacity for non-English languages due to data contamination in the training 
data, but it is likely not to perform well.

---

"""

CONVO_INTRO_TEXT = """\
## Conversation content

Each line is an utterance in the conversation to generate. Speakers alternate between A and B, starting with speaker A.
"""

DEFAULT_CONVERSATION = """\
Hey how are you doing.
Pretty good, pretty good.
I'm great, so happy to be speaking to you.
Me too, this is some cool stuff huh?
Yeah, I've been reading more about speech generation, and it really seems like context is important.
Definitely.
"""

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": "prompts/conversational_a.wav",
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": "prompts/conversational_b.wav",
    },
    "read_speech_a": {
        "text": (
            "And Lake turned round upon me, a little abruptly, his odd yellowish eyes, a little "
            "like those of the sea eagle, and the ghost of his smile that flickered on his "
            "singularly pale face, with a stern and insidious look, confronted me."
        ),
        "audio": "prompts/read_speech_a.wav",
    },
    "read_speech_b": {
        "text": (
            "He was such a big boy that he wore high boots and carried a jack knife. He gazed and "
            "gazed at the cap, and could not keep from fingering the blue tassel."
        ),
        "audio": "prompts/read_speech_b.wav",
    },
    "read_speech_c": {
        "text": (
            "All passed so quickly, there was so much going on around him, the Tree quite forgot "
            "to look to himself."
        ),
        "audio": "prompts/read_speech_c.wav",
    },
    "read_speech_d": {
        "text": (
            "Suddenly I was back in the old days Before you felt we ought to drift apart. It was "
            "some trick-the way your eyebrows raise."
        ),
        "audio": "prompts/read_speech_d.wav",
    },
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = hf_hub_download(repo_id="audo/csm-1b", filename="ckpt.pt")
generator = load_csm_1b(model_path, device)


@spaces.GPU(duration=gpu_timeout)
def infer(
    text_prompt_speaker_a,
    text_prompt_speaker_b,
    audio_prompt_speaker_a,
    audio_prompt_speaker_b,
    gen_conversation_input,
) -> tuple[np.ndarray, int]:
    # Estimate token limit, otherwise failure might happen after many utterances have been generated.
    if len(gen_conversation_input.strip() + text_prompt_speaker_a.strip() + text_prompt_speaker_b.strip()) >= 2000:
        raise gr.Error("Prompts and conversation too long.", duration=30)

    try:
        return _infer(
            text_prompt_speaker_a,
            text_prompt_speaker_b,
            audio_prompt_speaker_a,
            audio_prompt_speaker_b,
            gen_conversation_input,
        )
    except ValueError as e:
        raise gr.Error(f"Error generating audio: {e}", duration=120)


def _infer(
    text_prompt_speaker_a,
    text_prompt_speaker_b,
    audio_prompt_speaker_a,
    audio_prompt_speaker_b,
    gen_conversation_input,
) -> tuple[np.ndarray, int]:
    audio_prompt_a = prepare_prompt(text_prompt_speaker_a, 0, audio_prompt_speaker_a)
    audio_prompt_b = prepare_prompt(text_prompt_speaker_b, 1, audio_prompt_speaker_b)

    prompt_segments: list[Segment] = [audio_prompt_a, audio_prompt_b]
    generated_segments: list[Segment] = []

    conversation_lines = [line.strip() for line in gen_conversation_input.strip().split("\n") if line.strip()]
    for i, line in enumerate(conversation_lines):
        # Alternating speakers A and B, starting with A
        speaker_id = i % 2

        audio_tensor = generator.generate(
            text=line,
            speaker=speaker_id,
            context=prompt_segments + generated_segments,
            max_audio_length_ms=30_000,
        )
        generated_segments.append(Segment(text=line, speaker=speaker_id, audio=audio_tensor))

    # Concatenate all generations and convert to 16-bit int format
    audio_tensors = [segment.audio for segment in generated_segments]
    audio_tensor = torch.cat(audio_tensors, dim=0)

    # This applies an imperceptible watermark to identify audio as AI-generated.
    # Watermarking ensures transparency, dissuades misuse, and enables traceability.
    # Please be a responsible AI citizen and keep the watermarking in place.
    # If using CSM 1B in another application, use your own private key and keep it secret.
    audio_tensor, wm_sample_rate = watermark(
        generator._watermarker, audio_tensor, generator.sample_rate, CSM_1B_HF_WATERMARK
    )
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=wm_sample_rate, new_freq=generator.sample_rate
    )

    audio_array = (audio_tensor * 32768).to(torch.int16).cpu().numpy()

    return generator.sample_rate, audio_array


def prepare_prompt(text: str, speaker: int, audio_path: str) -> Segment:
    audio_tensor, _ = load_prompt_audio(audio_path)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)


def load_prompt_audio(audio_path: str) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    if sample_rate != generator.sample_rate:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor, orig_freq=sample_rate, new_freq=generator.sample_rate
        )
    return audio_tensor, generator.sample_rate


def create_speaker_prompt_ui(speaker_name: str):
    speaker_dropdown = gr.Dropdown(
        choices=list(SPEAKER_PROMPTS.keys()), label="Select a predefined speaker", value=speaker_name
    )
    with gr.Accordion("Or add your own voice prompt", open=False):
        text_prompt_speaker = gr.Textbox(label="Speaker prompt", lines=4, value=SPEAKER_PROMPTS[speaker_name]["text"])
        audio_prompt_speaker = gr.Audio(
            label="Speaker prompt", type="filepath", value=SPEAKER_PROMPTS[speaker_name]["audio"]
        )

    return speaker_dropdown, text_prompt_speaker, audio_prompt_speaker


with gr.Blocks() as app:
    gr.Markdown(SPACE_INTRO_TEXT)
    gr.Markdown("## Voices")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Speaker A")
            speaker_a_dropdown, text_prompt_speaker_a, audio_prompt_speaker_a = create_speaker_prompt_ui(
                "conversational_a"
            )

        with gr.Column():
            gr.Markdown("### Speaker B")
            speaker_b_dropdown, text_prompt_speaker_b, audio_prompt_speaker_b = create_speaker_prompt_ui(
                "conversational_b"
            )

    def update_audio(speaker):
        if speaker in SPEAKER_PROMPTS:
            return SPEAKER_PROMPTS[speaker]["audio"]
        return None

    def update_text(speaker):
        if speaker in SPEAKER_PROMPTS:
            return SPEAKER_PROMPTS[speaker]["text"]
        return None

    speaker_a_dropdown.change(fn=update_audio, inputs=[speaker_a_dropdown], outputs=[audio_prompt_speaker_a])
    speaker_b_dropdown.change(fn=update_audio, inputs=[speaker_b_dropdown], outputs=[audio_prompt_speaker_b])

    speaker_a_dropdown.change(fn=update_text, inputs=[speaker_a_dropdown], outputs=[text_prompt_speaker_a])
    speaker_b_dropdown.change(fn=update_text, inputs=[speaker_b_dropdown], outputs=[text_prompt_speaker_b])

    gr.Markdown(CONVO_INTRO_TEXT)

    gen_conversation_input = gr.TextArea(label="conversation", lines=20, value=DEFAULT_CONVERSATION)
    generate_btn = gr.Button("Generate conversation", variant="primary")
    gr.Markdown("GPU time limited to 3 minutes, for longer usage duplicate the space.")
    audio_output = gr.Audio(label="Synthesized audio")

    generate_btn.click(
        infer,
        inputs=[
            text_prompt_speaker_a,
            text_prompt_speaker_b,
            audio_prompt_speaker_a,
            audio_prompt_speaker_b,
            gen_conversation_input,
        ],
        outputs=[audio_output],
    )

app.launch(ssr_mode=True)
