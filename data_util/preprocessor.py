
def construct_context(fin_path="../data/corpus.test.nolabel.txt",
                      fout_path="../data/corpus.test.context.txt"):

    last_line = ""

    with open(fout_path, "w") as fout:
        with open(fin_path) as fin:
            for cur_line in fin:

                if cur_line.strip() == "":
                    last_line = ""
                    fout.write("\n")
                    continue

                line_seg = cur_line.strip().lower().split("\t")
                id = line_seg[0]
                text = line_seg[1]

                if last_line == "":
                    context = "NULL"
                else:
                    context = last_line

                last_line = text

                fout.write(id)
                fout.write("\t")
                fout.write(text)
                fout.write("\t")
                fout.write(context)
                fout.write("\n")


def check_is_chinese(ch):
    if u'\u4e00' <= ch <= u'\u9fff':
        return True
    return False


def whitespece_segment(input_text):
    text_list = input_text.strip()

    seg_text = []
    for idx in range(len(text_list) - 1):
        if check_is_chinese(text_list[idx]) or check_is_chinese(text_list[idx + 1]):
            seg_text.append(text_list[idx])
            seg_text.append(" ")
        else:
            seg_text.append(text_list[idx])
    seg_text.append(text_list[-1])
    text = "".join(seg_text)

    return text


if __name__ == "__main__":
    construct_context()