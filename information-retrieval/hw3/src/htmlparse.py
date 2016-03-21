import html2text


def parse_html(content):
    h2t = html2text.HTML2Text()
    h2t.ignore_links = True
    h2t.ignore_images = True
    h2t.images_to_alt = False

    return h2t.handle(content)
