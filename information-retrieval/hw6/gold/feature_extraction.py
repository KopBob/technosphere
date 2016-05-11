import re


class Object:
    def __init__(self, pos, punctuation_char, is_data_point, label, text_num=None):
        self.text_num = text_num
        self.pos = pos
        self.punctuation_char = punctuation_char
        self.is_data_point = is_data_point
        self.label = label

    def __repr__(self):
        return str(self.__dict__)


def extract_features(data_points, paragraph, include_extra=True):
    for i, point in enumerate(data_points):
        if point.is_data_point:
            features = {}

            prev_point = data_points[i - 1]
            next_point = data_points[i + 1]

            prev_text = paragraph[prev_point.pos + 1:point.pos]
            next_text = paragraph[point.pos + 1:next_point.pos]

            prev_text_split = prev_text.split()
            next_text_split = next_text.split()

            prev_word = prev_text_split[-1] if len(prev_text_split) else ''
            next_word = next_text_split[0] if len(next_text_split) else ''

            features["punctuation_kind"] = point.punctuation_char
            features["prev_punctuation_kind"] = prev_point.punctuation_char
            features["next_punctuation_kind"] = next_point.punctuation_char

            features["dist_to_prev"] = point.pos - prev_point.pos
            features["dist_to_next"] = next_point.pos - point.pos

            features["len_of_prev_word"] = len(prev_word)
            features["len_of_next_word"] = len(next_word)

            features["is_prev_uppercase"] = sum([c.isupper() for c in prev_word])
            features["is_next_uppercase"] = sum([c.isupper() for c in next_word])

            # if include_extra:
            features["_label"] = point.label
            features["_text_num"] = point.text_num
            features["_pos"] = point.pos

            yield features


def extract_features_from_labeled_data(paragraphs, paragraph_sentences):
    for paragraph_num, sentences in enumerate(paragraph_sentences):
        curr_pos = -1

        start_obj = Object(text_num=paragraph_num, pos=curr_pos,
                           punctuation_char='|', is_data_point=0, label=1)
        dividers = [start_obj]

        for sentence in sentences:
            curr_pos += 1  # consider last char

            pieces = re.split('(\.|\?|\!)', sentence)
            if not pieces[-1]:  # Removing extra part
                pieces = pieces[:-1]

            # Marking all dividers in sentence
            for i, piece in enumerate(pieces):
                if re.match('(\.|\?|\!)', piece):
                    obj = Object(text_num=paragraph_num, pos=curr_pos,
                                 punctuation_char=piece, is_data_point=1, label=1)
                    dividers.append(obj)

                curr_pos += len(piece)

            # Last divider in sentence will be marked as -1
            # Other dividers will be marked as 1
            if dividers[-1].label:
                dividers[-1].label = -1

        # Don't consider last divider in paragraph as data point
        if dividers[-1].label:
            dividers[-1].is_data_point = 0

        yield list(extract_features(dividers, paragraphs[paragraph_num]))


def extract_features_from_unlabeled_data(paragraph):
    curr_pos = -1

    start_obj = Object(pos=curr_pos,
                       punctuation_char='|', is_data_point=0, label=0)
    dividers = [start_obj]

    pieces = re.split('(\.|\?|\!)', paragraph)
    if not pieces[-1]:  # Removing extra part
        pieces = pieces[:-1]

    curr_pos += 1

    is_last_punct = False
    for i, piece in enumerate(pieces):
        if re.match('(\.|\?|\!)', piece):
            obj = Object(pos=curr_pos, punctuation_char=piece, is_data_point=1, label=0)

            dividers.append(obj)
            is_last_punct = True
        else:
            is_last_punct = False

        curr_pos += len(piece)

    if is_last_punct:
        dividers[-1].is_data_point = 0
    else:
        end_obj = Object(pos=curr_pos, punctuation_char='.', is_data_point=0, label=0)

        dividers.append(end_obj)

    return list(extract_features(dividers, paragraph, include_extra=False))
