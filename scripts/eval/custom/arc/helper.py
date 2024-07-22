def arc_prompt(input):
    question = input['question']
    choices_text = input['choices']['text']
    choices_label = input['choices']['label']
    choices_formatted = "\n".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])

    return f"User: {question} Choose the best option from below:\n{choices_formatted}\nAssistant: The best answer is\""

def arc_prompt_llava(input):
    question = input['question']
    choices_text = input['choices']['text']
    choices_label = input['choices']['label']
    choices_formatted = ", ".join([f"{label}. {text}" for label, text in zip(choices_label, choices_text)])

    return f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: Question: {question}\nOptions: {choices_formatted}\nAnswer with the option's letter from the given choices directly. ASSISTANT:"