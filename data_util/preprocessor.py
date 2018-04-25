
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


if __name__ == "__main__":
    construct_context()