from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class PredictionPipeline:

    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=8,
            max_length=128,
            length_penalty=0.8,
            early_stopping=True
        )

        output = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        print("Dialogue:")
        print(text)

        print("\nModel Summary:")
        print(output)

        return output