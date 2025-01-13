from qwen_vl_utils import process_vision_info
from transformers import TextIteratorStreamer
from concurrent.futures import ThreadPoolExecutor
import time
import torch 

from chat import transform_image
from chat_template_llm import Assistant as _Assistant, LongMemmoryAssistant as _LongMemmoryAssistant

from memory_retrieval import * 
from episodic_attention import * 

class Assistant(_Assistant): 
    @staticmethod
    def process_message_image(message, *args, **kwargs):
        if ('<image>' in message) and '</image>' in message:
            image_path = message.split('<image>')[-1].split('</image>')[0]
            transform_image (image_path, *args, **kwargs)

            message = [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text": ' '.join([
                message.split("<image>")[0],
                message.split("</image>")[-1]])},]
        return message

    def process_message(self, message):
        """
        Tokenize and prepare conversation inputs for the model.

        :param message: The user's input message.
        :return: Generation arguments and the streamer object.
        """
        # Process in if message has image inside
        message = self.process_message_image(message)

        # Add the user message to the conversation
        self.conversation.append({'role': 'user', 'content': message})
        conversation_template = self.processor.tokenizer.apply_chat_template(self.conversation, tokenize=False)

        conversation_text = conversation_template + "\n<|im_start|>assistant\n"

        streamer = TextIteratorStreamer(self.processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        image_inputs, video_inputs = process_vision_info(self.conversation)
        inputs = self.processor(
            text=[conversation_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")


        generation_kwargs = dict(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15,
            streamer=streamer,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        return generation_kwargs, streamer

class LongMemmoryAssistant(_LongMemmoryAssistant): 
    def __init__(self, episodic_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodic_config = episodic_config
        self.executor = ThreadPoolExecutor(max_workers=1)  # For background execution

    def remember_for_long(self, message_text: str):

        if "<image>" in message_text:
            message = self.process_message_image(message_text, to_size=128)

            image_path = [i for i in message if i['type'] == 'image'][0]['image']

            messages = [{'role': 'user', 'content': [
                {'type': 'image', 'image': image_path},
                {'type': 'text', 'text': 'Detail this IMAGE in all details'}
                ]}]

            conversation_template = self.processor.tokenizer.apply_chat_template(messages, tokenize=False)

            conversation_text = conversation_template + "\n<|im_start|>assistant\n"
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[conversation_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            print(f'processing image {image_path}')
            with torch.no_grad():
                # Inference: Generation of the output
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

            message_text = message_text.split('<image>')[0] + f"[image: {output_text}]" + message_text.split('<image>')[-1]

        def process_remember(message):
            # Minimize GPU memory usage
            with torch.no_grad():
                surprise_scores, similarity_matrix, tokens = episodic_suprise_setup_v1(
                    message, self.model, self.processor, self.episodic_config
                )
                history_events, refined_boundaries = episodic_suprise_setup_v2(
                    surprise_scores, similarity_matrix, tokens, self.episodic_config
                )
                for event in history_events:
                    if len(event):
                        # Store event with timestamp
                        self.event_memory.store_event(event, time.strftime("%Y-%m-%d %H:%M"))

        # Submit the process_remember task to the executor
        self.executor.submit(process_remember, message_text)

        return None