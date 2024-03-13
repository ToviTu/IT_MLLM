# Prompt templates for LLaVA

llava_qa = lambda q: f"User: {q}\nAssistant:"
llava_cqa = lambda q, c: f"Context: {c}\nUser: {q}\nAssistant:"
llava_mc = lambda q, c: f"User: {q} Choose from below.\nChoices: A) {c[0]}\n B) {c[1]}\n C) {c[2]}\n D) {c[3]}\nAssistant:"
llava_vqa = lambda q, c: f"<image>{c}\nUser: {q}\nAssistant:"

