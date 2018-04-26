# -----------------------------------
#
#
#
# -----------------------------------

from rule_model.rule_matcher import RuleMatcher
from bilstm_model.bilstm_model import BiLSTMModel, BiLSTMModelContext
from data_util.output import ResultWriter
from data_util.preprocessor import whitespece_segment

class Config():

    # Configuration for Rule_Matcher
    jieba_user_dict = "./dicts/user_dict.txt"
    song_list = "./dicts/song.txt"
    singer_list = "./dicts/singer.txt"
    keywords_dict = "./rule_model/keyword.json"
    command_json_path = "./rule_model/command.json"
    ngram_json_path = "./rule_model/ngram.json"

    # Configuration for BiLSTM_Model
    non_context_bilstm_data_path = "./bilstm_model/data/non-context"
    non_context_bilstm_model_path = "./bilstm_model/model/non-context"
    context_bilstm_data_path = "./bilstm_model/data/context"
    context_bilstm_model_path = "./bilstm_model/model/context"
    modes = ["non-context", "single_context", "fully_context"]
    model_type = "non-context"

    # Input files
    input_fpath = "./data/corpus.test.context.txt"

    # Output files
    output_fpath = "./result/output.txt"

    slot_intents = ["music.play", "navigation.navigation","phone_call.make_a_phone_call"]

def extract_song_slot(text, context):
    text_seg = whitespece_segment(text)

    slots = ["B-song"]
    for _ in range(len(text_seg.strip().split(" "))-1):
        slots.append("I-song")

    return slots

def extract_number_slone(text, context):
    text_seg = whitespece_segment(text)

    assert len(text_seg) == 1

    return ["B-phone_num"]

if __name__ == "__main__":

    config = Config()

    if config.model_type not in config.modes:
        print("Invalid Model Type. Must be one of [non-context, single_context, fully_context].")
        exit(1)

    rule_matcher = RuleMatcher(config.jieba_user_dict, config.keywords_dict, config.command_json_path, config.ngram_json_path, config.song_list)

    if config.model_type == "non-context":
        bi_lstm = BiLSTMModel(config.non_context_bilstm_data_path, config.non_context_bilstm_model_path)
    else:
        bi_lstm = BiLSTMModelContext(config.context_bilstm_data_path, config.context_bilstm_model_path)

    rw = ResultWriter(config)

    last_intent = None

    with open(config.input_fpath) as fin:
        for line in fin:
            line_seg = line.strip().split("\t")

            if len(line_seg) > 1:
                id, text, context = line_seg
                # 首先使用三层规则匹配
                intent, pattern = rule_matcher.match(text, last_intent)
                # 使用Bi-LSTM模型匹配
                if intent is None:
                    intent = bi_lstm.predict_intent(text, context)

                if intent in Config.slot_intents:
                    if pattern == "#SONGMATCH":
                        slots = extract_song_slot(text, context)
                    elif pattern == "#NUMBERMATCH":
                        slots = extract_song_slot(text, context)
                    else:
                        slots = bi_lstm.extract_slot(text, context)
                else:
                    slots = bi_lstm.extract_non_slot(text)

                rw.save_line(id, text, intent, slots)
                if intent != 'OTHERS':
                    last_intent = intent

            else:
                rw.save_blank_line()
                last_intent = None

