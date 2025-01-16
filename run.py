
from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoModelForCausalLM,
                          AutoProcessor,
                          AutoModelForImageTextToText)
from memory_retrieval import MemoryDatabase, EventMemory
import torch

import yaml
import time


def fill_memory(event_memory): 

    if len(event_memory.memory) < 10:

        print('[Memory is empty, Adding Irelevant info to memory]')
        some_info = [
            "The unexamined life is not worth living. – Socrates",
            "Happiness is not an ideal of reason but of imagination. – Immanuel Kant",
            "You are not a drop in the ocean. You are the entire ocean in a drop. – Rumi",
            "What we achieve inwardly will change outer reality. – Plutarch",
            "Man is condemned to be free; because once thrown into the world, he is responsible for everything he does. – Jean-Paul Sartre",
            "In the middle of difficulty lies opportunity. – Albert Einstein",
            "Time is an illusion. Lunchtime doubly so. – Douglas Adams",
            "The only way to deal with an unfree world is to become so absolutely free that your very existence is an act of rebellion. – Albert Camus",
            "The greatest wealth is to live content with little. – Plato",
            "We are what we repeatedly do. Excellence, then, is not an act, but a habit. – Aristotle"
        ]


        for i in some_info:
            event_memory.store_event(i, time.strftime("%Y-%m-%d %H:%M"))
        print("[Memory filled]")

def run_chat(): 
    with open("config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    model_id = config['model_id']
    tokenizer_id = config['tokenizer_id']

    if config['session_type'] == 'llm':

        class Processor:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(self, *args, **kwargs):
                return self.tokenizer( *args, **kwargs)

        processor = Processor(AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True))

        device_0 = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_0,
            use_cache=False,
            trust_remote_code=True,
            )

    elif config['session_type'] == 'vllm':
        processor = AutoProcessor.from_pretrained(model_id)

        device_0 = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map=device_0,
            use_cache=False,
            trust_remote_code=True,
            )

    # model1 = AutoModelForImageTextToText.from_pretrained(
    #     model_id,
    #     device_map=device_0,
    #     use_cache=True,
    #     trust_remote_code=True,
    #     )
    memory = MemoryDatabase(config['memories_dir'])

    print("[Memories Loaded]")

    emb_model = AutoModel.from_pretrained(config['embedding_model_id'])
    emb_tokenizer = AutoTokenizer.from_pretrained(config['embedding_tokenizer_id'])

    event_memory = EventMemory(emb_model, emb_tokenizer, memory)
    fill_memory(event_memory)

    if config['session_type'] == 'llm':

        from chat_template_llm import LongMemmoryAssistant

    elif config['session_type'] == 'vllm':

        from chat_template_vllm import LongMemmoryAssistant

        # (Episodic Config)
    class EpisodicSupriseSetupConfig:
        batch_size = config['batch_size']
        scaling_factor  = config['variance_scaling_factor']
        threshold = None if config['threshold'] == "None" else config['threshold']

        # Needed arguments for chat:
        # Episodic_config,
        # processor
        # model,
        # event_memory,
        # k_memmories
        # top_k_memmories

    chat = LongMemmoryAssistant(EpisodicSupriseSetupConfig(),
                                    processor,
                                    model,
                                    event_memory,
                                    k_memmories = config['k_memmories'],
                                    top_k_memmories = config['k_top_k_memmories']
                                    )
    chat.start_conversation()

if __name__ == "__main__": 
    run_chat() 