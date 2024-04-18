# Prompt templates for LLaVA

llava_qa = lambda q, a: f"User: {q}\nAssistant: {a}"
llava_cqa = lambda c, q, a: f"Context: {c}\nUser: {q}\nAssistant: {a}"
llava_mc = lambda c, q, a: f"User: {q} Choose from below.\nChoices: A) {c[0]}\n B) {c[1]}\n C) {c[2]}\n D) {c[3]}\nAssistant: {a}"
llava_vqa = lambda c, q, a: f"<image>{c}\nUser: {q}\nAssistant:{a}"

# Prompt templates for Other models

llm_qa = lambda q, a: f"Question: {q}\nAnswer: {a}"
llm_cqa = lambda c, q, a: f"Context: {c}\nQuestion: {q}\nAnswer: {a}"
llm_mc = lambda c, q, a: f"Question: {q} Choose from below.\nChoices: A) {c[0]}\n B) {c[1]}\n C) {c[2]}\n D) {c[3]}\nAnswer: {a}"
llm_vqa = lambda c, q, a: f"<image>{c}\nQuesiton: {q}\nAnswer:{a}"

