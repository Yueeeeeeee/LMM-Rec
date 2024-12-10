import json
from copy import deepcopy

from constants import *


def construct_item_text(args,
                        processor,
                        item,
                        meta,
                        image_dict,
                        image_paths,
                        ):
    item_attrs = []
    if item in image_dict:
        image_paths.append(image_dict[item])
        item_attrs.append('"image": ' + DEFAULT_IMAGE_TOKEN)

    for key in args.lmm_text_attributes:
        if key not in meta: continue

        attr = meta[key]
        numerical = False
        if isinstance(attr, list):
            attr = ", ".join(attr[0])
        elif isinstance(attr, float) or isinstance(attr, int):
            attr = str(attr)
            numerical = True
        attr = processor.tokenizer.tokenize(attr)[:args.lmm_max_attr_len]
        if numerical:
            item_attrs.append('"' + key + '": ' + processor.tokenizer.convert_tokens_to_string(attr))
        else:
            item_attrs.append('"' + key + '": "' + processor.tokenizer.convert_tokens_to_string(attr) + '"')

    return json.dumps("{" + ", ".join(item_attrs) + "}")[1:-1]


def prepare_multimodal_input(args,
                             seq,
                             candidates,
                             label,
                             meta_dict,
                             image_dict,
                             processor,
                             eval=False
                             ):
    image_paths = []
    seq_t = "\n".join([construct_item_text(
        args, processor, item, meta_dict[item], image_dict, image_paths) for item in seq])
    can_t = "\n".join(["(" + chr(ord("A") + idx) + ") " + construct_item_text(
        args, processor, item, meta_dict[item], image_dict, image_paths) for idx, item in enumerate(candidates)])
    output = chr(ord("A") + candidates.index(label))

    raw_prompt = args.lmm_instruct_template.format(seq_t, can_t)
    split_prompt = raw_prompt.split(DEFAULT_IMAGE_TOKEN)
    image_paths = [str(x) for x in image_paths] + [None]

    multimodal_content = []
    for text, image in zip(split_prompt, image_paths):
        multimodal_content.append({"type": "text", "text": text})
        if image is not None:
            multimodal_content.append({"type": "image", "image": image})

    messages = [
        {
            "role": "user",
            "content": multimodal_content,
        }
    ]
    eval_prompt = processor.apply_chat_template(
        messages, tokenize=False, 
        add_generation_prompt=True,
    )

    full_messages = deepcopy(messages)
    full_messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": output}
            ]
        }
    )
    full_prompt = processor.apply_chat_template(
        full_messages, tokenize=False, 
        add_generation_prompt=False,
    )

    if eval:
        return {
            "text": eval_prompt,
            "message": messages,
            "labels": ord(output) - ord("A"),
        }
    else:
        return {
            "text": full_prompt,
            "message": messages,
            "eval_text": eval_prompt,  # for label generation
        }