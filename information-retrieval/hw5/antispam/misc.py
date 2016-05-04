import re


def unicode_filter(unicode_range="\u0400-\u0500"):
    if len(arr) == 0: return []
    text = " ".join(arr)
    russian_words = re.findall(u"[%s]+" % unicode_range, text.lower())
    return [w for w in russian_words if w not in STOP_WORDS]