from __future__ import annotations


def build_prompt_with_images(processor, images, question: str) -> str:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
