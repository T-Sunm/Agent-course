from src.tools.vqa_tool import caption_image


def caption_node(state):
        caption = caption_image(state["image"])
        return {"image_caption": caption}