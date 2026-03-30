
def build_user_message(prompt, image_list=None):
    """
    构建用户消息，支持文本和图片
    """
    from .image_utils import prepare_base64_image

    if image_list and len(image_list) > 0:
        processed_images = []
        for img in image_list:
            processed_img = prepare_base64_image(img)
            if processed_img:
                processed_images.append(processed_img)

        if processed_images:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in processed_images]
                ]
            }

    return {"role": "user", "content": prompt}



def build_image_instruction(imgs):
    lines = []
    img_idx = 1

    for q_idx, group in enumerate(imgs, start=1):
        if not group:
            lines.append(f"第{q_idx}题：无图片")
        else:
            indices = list(range(img_idx, img_idx + len(group)))
            lines.append(
                f"第{q_idx}题：只能使用图片 {indices}"
            )
            img_idx += len(group)
    return "\n".join(lines)