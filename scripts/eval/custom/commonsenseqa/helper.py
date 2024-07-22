from llava.conversation import conv_templates


def prompt(input):
    question = input['question']
    choices_text = input['choices']['text']
    choices_label = input['choices']['label']
    choices_formatted = "\n".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])

    return f"Question: {question} Choose the best option from below:\n{choices_formatted}\nThe best answer is\""
    #return f"User: {question} Choose the best option from below:\n{choices_formatted}\n\nAssistant: The best answer is"

def lit_prompt(input):
    question = input['question']
    choices_text = input['choices']['text']
    choices_label = input['choices']['label']
    choices_formatted = ", ".join([f"{label}: {text}" for label, text in zip(choices_label, choices_text)])
    prompt = f"Question: {question}\nOptions: {choices_formatted}\n Answer with the option's letter from the given choices directly."
    conv = conv_templates['v1'].copy()

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    final_prompt = conv.get_prompt()

    return final_prompt