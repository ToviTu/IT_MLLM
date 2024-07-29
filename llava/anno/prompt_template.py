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
    "Choose the correct option from the following:",
    "Select the most appropriate answer from the following options:",
    "Pick the correct answer from the given choices:",
    "Identify the correct option among the following:",
    "Select the right answer from the options below:",
    "Choose the best answer from the following choices:",
    "Which option is correct?",
    "Select the accurate answer from the options provided:",
    "Choose the correct response from the given options:",
    "From the options below, pick the correct answer:",
    "Which of these is the correct answer?",
    "Choose the most accurate answer from the following options:",
    "Select the appropriate answer from the given choices:",
    "Identify the correct answer from the options below:",
    "Pick the right option from the following:",
    "Choose the best option from the choices provided:",
    "Select the correct answer from these options:"
]

short_answer_prompt = [
    "Please provide a short answer to the question:",
    "Please answer the question in a few words:",
    "Please provide a brief answer to the question:",
    "Please answer the question briefly:",
    "Please provide a concise answer to the question:",
    "Give a short response to the question:",
    "Answer the question in one sentence:",
    "Respond to the question with a brief statement:",
    "Provide a succinct answer to the question:",
    "Answer the question concisely:",
    "Give a brief answer to the following:",
    "Respond briefly to the question:",
    "Provide a short response:",
    "Please reply with a concise answer:",
    "Answer with a brief statement:",
    "Give a succinct response:",
    "Respond in a few words:",
    "Provide a concise reply:",
    "Answer shortly and precisely:",
    "Give a brief and accurate answer:"
]

fa_prompt = [
    "To sum up, the direct answer is:",
    "To summarize, the direct answer is:",
    "In summary, the direct answer is:",
    "To conclude, the direct answer is:",
    "In closing, the direct answer is:",
    "All in all, the direct answer is:",
    "Overall, the direct answer is:",
    "In brief, the direct answer is:",
    "In short, the direct answer is:",
    "In essence, the direct answer is:",
    "To encapsulate, the direct answer is:",
    "In a nutshell, the direct answer is:",
    "To wrap up, the direct answer is:",
    "As a final point, the direct answer is:",
    "Ultimately, the direct answer is:",
    "As a final note, the direct answer is:",
    "In the end, the direct answer is:",
    "To recapitulate, the direct answer is:",
    "To put it briefly, the direct answer is:",
    "To round off, the direct answer is:"
]

tf_prompt = [
    "Please answer the question with 'yes' or 'no'.",
    "Respond to the question with 'yes' or 'no'.",
    "Answer the question using 'yes' or 'no'.",
    "Provide a 'yes' or 'no' answer to the question.",
    "Reply to the question with 'yes' or 'no'.",
    "Use 'yes' or 'no' to answer the question.",
    "Give a 'yes' or 'no' response to the question.",
    "Please respond with 'yes' or 'no'.",
    "Your answer should be either 'yes' or 'no'.",
    "Answer with 'yes' or 'no'.",
    "Please provide a 'yes' or 'no' answer.",
    "Reply using 'yes' or 'no'.",
    "Respond using 'yes' or 'no'.",
    "Give an answer of 'yes' or 'no'.",
    "Use 'yes' or 'no' in your response.",
    "Please use 'yes' or 'no' to answer.",
    "Provide a response of 'yes' or 'no'.",
    "Answer simply with 'yes' or 'no'.",
    "Respond solely with 'yes' or 'no'.",
    "Your response should be 'yes' or 'no'."
]



# Prompt templates for LLaVA

llava_cqa = lambda c, q, a: "User: " + (f'Context: {c}\n' if c else "") + q + "\nAssistant: " + a
llava_srq = lambda c, q, a: llava_cqa(c, random.choice(short_answer_prompt) + '\n'+ q, a)
llava_tfq = lambda c, q, a: llava_cqa(c, random.choice(tf_prompt) + '\n'+ q, a)
llava_cmc = lambda c, mc, q, a: llava_cqa(
    c,
    q+" "+random.choice(mc_prompt)+"\n"+"\n".join([each for each in mc]),
    a
)

# Prompt templates for Other models

llm_qa = lambda q, a: f"Question: {q}\nAnswer: {a}"
llm_cqa = lambda c, q, a: f"Context: {c}\nQuestion: {q}\nAnswer: {a}"
llm_mc = lambda c, q, a: f"Question: {q} Choose from below.\nChoices: A) {c[0]}\n B) {c[1]}\n C) {c[2]}\n D) {c[3]}\nAnswer: {a}"
llm_vqa = lambda c, q, a: f"<image>{c}\nQuesiton: {q}\nAnswer:{a}"

