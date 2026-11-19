# WHAT: Records how Hugging Face's `tokenizers` encodes a few strings with a Llama-3 `tokenizer.json`, so
#       Mamba3LiveTests checks Frigate's BPETokenizer (its `ignore_merges` path included) against an independent
#       implementation. Ran once; the fixture is committed (Tests/FrigateTests/Fixtures/llama3-tokens.json).
#   python3 -I scripts/llama3-token-fixture.py <folder with tokenizer.json> > Tests/FrigateTests/Fixtures/llama3-tokens.json
import json, sys
from tokenizers import Tokenizer

folder = sys.argv[1]
tokenizer = Tokenizer.from_file(f"{folder}/tokenizer.json")
texts = [
    "The history of the city begins",
    "func read(_ tokens: MLXArray, cache: [KVCache]) -> Mamba3Reading {\n    let h = backbone.embed(tokens)\n}",
    "    if let ssm, let angle0 { return SSD3.wrap(angle0 + increments) }  // carried\n",
    "Ich möchte über die Straße gehen — naïve café, 日本語のテキスト, emoji 🧵🪡.",
    "user@example.com wrote: \"don't\" isn't won't; 1,234,567.89 and 0x1F600; ___init___ __main__",
    "        \t\t  trailing spaces and tabs   \n\n\n",
    "ThisIsAVeryLongIdentifierNameWithCamelCaseAndNumbers123456789AndMoreWordsToMerge",
]
json.dump({"tokenizer": json.load(open(f"{folder}/tokenizer.json"))["model"].get("ignore_merges"),
           "cases": [{"text": t, "ids": tokenizer.encode(t, add_special_tokens=True).ids} for t in texts]},
          sys.stdout, ensure_ascii=False, indent=1)
