
def save_output(fout, id, text, intent, slots):
    fout.write(id)
    fout.write("\t")
    fout.write(text)
    fout.write("\t")
    fout.write(intent)
    fout.write("\t")
    fout.write(" ".join(slots))
    fout.write("\n")


def save_analysis_rulematch(fout, id, text, intent, slots):
    fout.write(id)
    fout.write("\t")
    fout.write(text)
    fout.write("\t")
    fout.write(intent)
    fout.write("\t")
    fout.write(" ".join(slots))
    fout.write("\n")

def save_analysis_bilistm(fout, id, text, intent, slots):
    fout.write(id)
    fout.write("\t")
    fout.write(text)
    fout.write("\t")
    fout.write(intent)
    fout.write("\t")
    fout.write(" ".join(slots))
    fout.write("\n")