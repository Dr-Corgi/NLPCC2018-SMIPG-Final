# -----------------------------------
#
#
#
# -----------------------------------

from rule_model.rule_matcher import RuleMatcher
from bilstm_model.bilstm_model import BiLSTMModel
from data_util.output import save_output, save_analysis_bilistm, save_analysis_rulematch

class Config():

    # Configuration for Rule_Matcher
    jieba_user_dict = "./rule_model/user_dict.txt"
    keywords_dict = "./rule_model/keywords_dict.txt"

    # Configuration for BiLSTM_Model
    bilstm_data_path = "./bilstm_model/data"
    bilstm_model_path = "./bilstm_model/model/non_context"

    # Input files
    input_fpath = "./data/corpus.test.context.txt"

    # Output files
    output_fpath = "./result/output.txt"


if __name__ == "__main__":

    rule_matcher = RuleMatcher(Config.jieba_user_dict, Config.keywords_dict)
    bi_lstm = BiLSTMModel(Config.bilstm_data_path, Config.bilstm_model_path)

    analysis_rule = open("./result/analysis_rule.txt", 'w')
    analysis_bilstm = open("./result/analysis_bilstm.txt",'w')

    with open(Config.input_fpath) as fin:
        with open(Config.output_fpath, 'w') as fout:
            for line in fin:
                line_seg = line.strip().split("\t")

                if len(line_seg) > 1:
                    id, text, context = line_seg

                    # 首先使用三层规则匹配
                    intent, pattern = rule_matcher.match(text)

                    # 使用Bi-LSTM模型匹配
                    if intent is None:
                        intent = bi_lstm.predict_intent(text)

                    if intent in ["music.play", "navigation.navigation","phone_call.make_a_phone_call"]:
                        slots = bi_lstm.extract_slot(text)
                    else:
                        slots = bi_lstm.extract_non_slot(text)

                    save_output(fout, id, text, intent, slots)

                    # Analysis
                    if pattern:
                        save_analysis_rulematch(analysis_rule, id, text, intent, slots)
                    else:
                        save_analysis_bilistm(analysis_bilstm, id, text, intent, slots)

                else:
                    fout.write("\n")

