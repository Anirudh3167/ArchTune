# datasets/registry.py

SUPPORTED_DATASETS = {
    "wiki-text-103": {
        "source": "huggingface",
        "url": "Salesforce/wikitext",
        "subset": "wikitext-103-v1",
        "preprocessing": "wiki-text-generation"
    },
    "openwebtext": {
        "source": "huggingface",
        "url": "vietgpt/openwebtext_en",
        "preprocessing": "text-generation"
    },
    "alpaca": {
        "source": "huggingface",
        "url" : "tatsu-lab/alpaca",
        "preprocessing": "chatbot"
    },
    "code-alpaca": {
        "source": "huggingface",
        "url": "flwrlabs/code-alpaca-20k",
        "preprocessing": "chatbot"
    },
    "causal-lm-instructions": {
        "source": "huggingface",
        "url": "causal-lm/instructions",
        "preprocessing": "chatbot"
    }
}
