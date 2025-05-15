import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.IndicTransToolkit import IndicProcessor
from flask import Flask, request, jsonify
import requests

# Initialize Flask app
app = Flask(__name__)

class TranslationService:
    def __init__(self, 
                 model_name="ai4bharat/indictrans2-en-indic-1B", 
                 src_lang="eng_Latn"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.src_lang = src_lang
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        self.preprocessor = IndicProcessor(inference=True)

    def translate(self, input_sentence, target_language):
        try:
            batch = self.preprocessor.preprocess_batch(
                [input_sentence],
                src_lang=self.src_lang,
                tgt_lang=target_language,
            )
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                )

            with self.tokenizer.as_target_tokenizer():
                translated_text = self.tokenizer.decode(
                    generated_tokens[0].detach().cpu(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )

            final_translation = self.preprocessor.postprocess_batch(
                [translated_text], 
                lang=target_language
            )[0]

            print(f"\n[INFO] Target Language: {target_language}")
            print(f"[INFO] Translated Text: {final_translation}\n")

            return final_translation
        except Exception as e:
            print(f"[ERROR] Translation failed: {str(e)}")
            return None

# Initialize the translator
translator = TranslationService()

@app.route('/')
def home():
    return jsonify({"message": "Translation Service is running."})

@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()

        if not data or 'text' not in data or 'target_language' not in data:
            return jsonify({'error': 'Request must contain `text` and `target_language` fields'}), 400

        translated_text = translator.translate(data['text'], data['target_language'])

        if not translated_text:
            return jsonify({'error': 'Translation failed.'}), 500

        # --- Make request to text-to-speech service ---
        try:
            payload = {
                "text": translated_text,
                "description": "Anu voice is warm and clear."
            }
            external_response = requests.post("http://127.0.0.1:1000/generate_audio", json=payload)
            external_response.raise_for_status()

            file_id = external_response.json().get('file_id')

            if not file_id:
                return jsonify({'error': 'No file_id received from TTS service.'}), 502

        except requests.RequestException as e:
            return jsonify({'error': f'Failed to communicate with TTS service: {str(e)}'}), 502

        return jsonify({
            'original_text': data['text'],
            'target_language': data['target_language'],
            'translated_text': translated_text,
            'file_id': file_id
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
