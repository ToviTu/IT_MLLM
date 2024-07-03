import random

# Prompt templates for Vicuna
vicuna_instruct = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
vicuna_qa = lambda q, a: f"{vicuna_instruct}USER: {q}\nASSISTANT: {a}"
vicuna_cqa = lambda c, q, a: f"{vicuna_instruct}{c}\nUSER: {q}\nASSISTANT: {a}"

# Multiple choice template

mc_prompt = [
    "Choose the correct answer from the following options:",
    "Only one of the following options is correct. Choose the correct one:",
    "Which of the following is the correct answer?",
    "Choose the correct option from the following:"
    "Select the most appropriate answer from the following options:",
]

fa_prompt = [
    "Therefore, the correct answer is:",
    "Hence, the correct answer is:",
    "To summarize, the correct answer is:",
    "In conclusion, the correct answer is:",
    "The best answer is:",
]

# Prompt templates for LLaVA

llava_cqa = lambda c, q, a: "User: " + (f'{c}\n' if c else '') + q + "\nAssistant: " + a
llava_vqa = lambda c, q, a: "<image>" + llava_cqa(c, q, a)
llava_cmc = lambda c, mc, q, a: llava_cqa(
    c,
    q+" "+random.choice(mc_prompt)+"\n".join([each for each in mc]),
    a
)

# Prompt templates for Other models

llm_qa = lambda q, a: f"Question: {q}\nAnswer: {a}"
llm_cqa = lambda c, q, a: f"Context: {c}\nQuestion: {q}\nAnswer: {a}"
llm_mc = lambda c, q, a: f"Question: {q} Choose from below.\nChoices: A) {c[0]}\n B) {c[1]}\n C) {c[2]}\n D) {c[3]}\nAnswer: {a}"
llm_vqa = lambda c, q, a: f"<image>{c}\nQuesiton: {q}\nAnswer:{a}"

