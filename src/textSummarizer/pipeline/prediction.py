import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class PredictionPipeline:

    def __init__(self):

        self.model_name = "Rupa-136/pegasus-samsum-model"
        self.subfolder = "pegasus-samsum-model"

        # Render Free / CPU
        self.device = torch.device("cpu")

        print("Loading Pegasus tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            subfolder=self.subfolder
        )

        print("Loading Pegasus model...")

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            subfolder=self.subfolder
        )

        self.model.to(self.device)
        self.model.eval()

        print("Pegasus model loaded successfully!")

    def predict(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        # Move inputs to CPU
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Disable gradient calculation during prediction
        with torch.no_grad():

            summary_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                max_new_tokens=128,
                length_penalty=0.8,
                early_stopping=True
            )

        output = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        print("Dialogue:")
        print(text)

        print("\nModel Summary:")
        print(output)

        return output