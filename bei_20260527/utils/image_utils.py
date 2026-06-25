import base64
from io import BytesIO

from bs4 import BeautifulSoup
from PIL import Image


def save_image_path(base64_str, save_path):
    """
    将 base64 字符串转换为图片并保存到指定路径

    Args:
        base64_str: base64 编码的图片字符串
        save_path: 保存的图片路径（目录路径，不包含文件名）

    Returns:
        str: 保存的完整文件路径
    """
    if base64_str.startswith('data:image/') and ';base64,' in base64_str:
        base64_str = base64_str.split(';base64,')[-1]

    base64_str = base64_str.replace('\n', '').replace(' ', '')
    missing_padding = len(base64_str) % 4
    if missing_padding:
        base64_str += '=' * (4 - missing_padding)

    image_data = base64.b64decode(base64_str, validate=True)

    image = Image.open(BytesIO(image_data))
    image.save(save_path)

    return save_path


def extract_image_from_html(html_tag):
    """
    从 HTML img 标签中提取 base64 图片数据

    Args:
        html_tag: 包含base64图片的HTML img标签（字符串或BeautifulSoup Tag对象）

    Returns:
        str: base64编码的图片字符串，失败返回None
    """
    if hasattr(html_tag, 'get'):
        src = html_tag.get('src', '')
        if src and src.startswith('data:image/'):
            src = src.replace("_x000d_", "")
            src = "".join(src.split())
            base64_str = src
            return base64_str
        else:
            return None


def is_valid_base64_image(encoded_image):
    if not isinstance(encoded_image, str) or len(encoded_image.strip()) == 0:
        return False

    try:
        if encoded_image.startswith('data:image/') and ';base64,' in encoded_image:
            encoded_image = encoded_image.split(';base64,')[-1]

        encoded_image = encoded_image.replace('\n', '').replace(' ', '')
        missing_padding = len(encoded_image) % 4
        if missing_padding:
            encoded_image += '=' * (4 - missing_padding)

        image_data = base64.b64decode(encoded_image, validate=True)
        if not image_data:
            return False

        with Image.open(BytesIO(image_data)) as image:
            image.verify()

        return True
    except (base64.binascii.Error,
            IOError, OSError,
            ValueError, TypeError):
        return False


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