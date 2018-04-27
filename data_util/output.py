from data_util.preprocessor import whitespece_segment
import re

class ResultWriter(object):

    def __init__(self, config):
        song_list = []
        singer_list = []

        with open(config.song_list) as fin:
            for line in fin:
                song = line.strip()
                song_list.append(song)
        self.song_list = sorted(song_list, key=lambda x: len(x), reverse=True)

        with open(config.singer_list) as fin:
            for line in fin:
                singer = line.strip()
                singer_list.append(singer)
        self.singer_list = sorted(singer_list, key=lambda x: len(x), reverse=True)

        self.fout = open(config.output_fpath, 'w')

    def fix(self, text, intent, slots):
        if intent == "music.play":
            # 首先设置为全0的SLOT
            fixed_slots = ["O" for _ in range(len(slots))]

            # 优先匹配歌名
            temp_text = text
            temp_index = 0
            for song in self.song_list:
                while(temp_text):
                    find_idx = temp_text.find(song)
                    if find_idx >= 0 and fixed_slots[find_idx] == 'O':
                        # 计算前面的长度
                        text_before = text[:find_idx]
                        if text_before == "":
                            text_before_len = 0 + temp_index
                        else:
                            text_before_len = len(whitespece_segment(text_before).split(" ")) + temp_index
                        # 计算歌名的长度
                        text_songname = song
                        song_len = len(whitespece_segment(text_songname).split(" "))
                        # 把这一段标记成歌曲
                        fixed_slots[text_before_len] = 'B-song'
                        for idx in range(text_before_len+1, text_before_len+song_len):
                            fixed_slots[idx] = 'I-song'

                        temp_text = temp_text[text_before_len+song_len:]
                        temp_index = text_before_len+song_len
                    else:
                        break

            # 其次匹配歌手
            temp_text = text
            temp_index = 0
            for singer in self.singer_list:
                while (temp_text):
                    find_idx = temp_text.find(singer)
                    if find_idx >= 0 and fixed_slots[find_idx] == 'O':
                        # 计算前面的长度
                        text_before = text[:find_idx]
                        if text_before == "":
                            text_before_len = 0 + temp_index
                        else:
                            text_before_len = len(whitespece_segment(text_before).split(" ")) + temp_index
                        # 计算歌名的长度
                        text_singername = singer
                        singer_len = len(whitespece_segment(text_singername).split(" "))
                        # 把这一段标记成歌曲
                        fixed_slots[text_before_len] = 'B-singer'
                        for idx in range(text_before_len + 1, text_before_len + singer_len):
                            fixed_slots[idx] = 'I-singer'

                        temp_text = temp_text[text_before_len + singer_len:]
                        temp_index = text_before_len + singer_len
                    else:
                        break

            slots = fixed_slots

        else:
            # print(text + '\t' + intent + '\t' + " ".join(slots))
            pass

        return self.slot_format(text, intent, slots)

    def slot_format(self, text, intent, slots):
        slot_text = []
        words = whitespece_segment(text).split(" ")
        slot_name = ""

        flag = 0
        need_fix_flag = False

        for i in range(len(words)):
            if slots[i] == 'O' and slot_name != '':
                if flag == 1:
                    slot_text.append("</" + slot_name + ">")
                slot_text.append(words[i])
                flag = 0
            elif slots[i] != 'O':
                if slots[i][0] == 'B':
                    if flag == 1 and slot_name != "" and intent != slots[i][2:]:
                        slot_text.append('</' + slot_name + '>')
                    flag = 1
                    slot_name = slots[i][2:]
                    slot_text.append('<' + slot_name + ">")
                slot_text.append(words[i])
                if (i == 0 and slots[i][0] == 'I') or (i > 0 and slots[i][0] == 'I' and slots[i - 1] == 'O'):
                    need_fix_flag = True
            elif slots[i] == 'O':
                slot_text.append(words[i])

        if flag != 0:
            slot_text.append('</'+slot_name+'>')

        return slot_text, need_fix_flag

    def save_line(self, id, text, intent, slots):
        slot_text, need_fix_flag = self.slot_format(text, intent, slots)

        if need_fix_flag:
            slot_text, need_fix_flag = self.fix(text, intent, slots)

        # fix contact_name
        regex = re.compile('<(?P<label>\S+)>(.*?)</(?P=label)>')
        tag = regex.findall(''.join(slot_text))
        for tag, word in tag:
            if tag == 'contact_name' and word.isdigit():
                slot_text = ''.join(slot_text).replace("contact_name", "phone_num")

        self.fout.write(id+"\t"+text+"\t"+intent+"\t"+"".join(slot_text)+"\n")

    def save_blank_line(self):
        self.fout.write("\n")