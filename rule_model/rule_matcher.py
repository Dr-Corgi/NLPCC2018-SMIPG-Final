import jieba
import json

class RuleMatcher(object):

    def __init__(self,
                 jieba_dict_path="./user_dict.txt",
                 dict_path="./keywords_dict.txt"):
        self.keywords = []

        self._load_jieba_dict(jieba_dict_path)
        self._load_keywords(dict_path)

    def _load_jieba_dict(self, jieba_dict_path):
        jieba.load_userdict(jieba_dict_path)

    def _load_keywords(self, dict_path):
        # 加载music.play高频关键词
        with open(dict_path) as fin:
            for line in fin:
                keyword = line.strip()
                self.keywords.append(keyword)

    def match(self, text):

        intent, pattern = self.sentence_match(text)

        if intent is None:
            intent, pattern = self.ngram_match(text)

        if intent is None:
            intent, pattern = self.keywords_match(text)

        # intent, pattern = self.keywords_match(text)

        return intent, pattern

    def sentence_match(self, text):
        sentence_pattern = {"music.next": ['换一首', '换一首歌','换一首音乐','换一首曲子'],
                            "music.play": ['我要听歌','音乐','播放音乐','放音乐','播放歌曲','搜歌曲','开始播放歌曲','继续播放','继续播放刚才的歌曲','放首音乐','放首曲子','放首歌','我要听','来一首歌'],
                            "music.pause": ['停止播放音乐','停止播放歌曲','暂停播放','关闭音乐','暂停音乐'],
                            "navigation.open": ['导航','打开导航','导航打开','我要导航'],
                            "navigation.start_navigation":['我要回家','开始导航','继续导航','恢复导航','接着导航','导航开始','导航继续','继续上次导航','继续上一次导航'],
                            "navigation.cancel_navigation": ['结束导航','导航取消','退出导航','导航结束','撤销导航','取消导航','关闭导航','导航关闭','导航退出'],
                            "phone_call.make_a_phone_call": ['打电话','拨打电话','我要打电话','打个电话','拨打10086'],
                            "phone_call.cancel": ['退出打电话','通话结束','不要打电话了','取消打电话'],
                            "OTHERS": ['叮当你好','好的','你好叮当','你好啊','打开蓝牙','连接蓝牙','关闭屏幕','打开收音机','打开行车记录仪','ok']}

        for key, value in sentence_pattern.items():
            for sent in value:
                if sent == text:
                    return key, sent

        return None, None

    def ngram_match(self, text):
        ngram_pattern = {"music.prev": ['上一首','上一曲'],
                         "music.next": ['更换歌曲','下一曲','下一首','后一首','切歌','换首歌','播放其他歌曲'],
                         "music.pause": ['歌曲暂停'],
                         "music.play": ['放一首','放一曲','唱一首','来首','来一首','点一首','请播放音乐'],
                         "navigation.open": ['打开导航','导航打开'],
                         "navigation.navigation": ['导航到','小学','大厦','收费站','怎么走','导航去','帮我导航到','小区','酒店','我要到'],
                         "navigation.start_navigation":['开始导航','继续导航','恢复导航','接着导航','导航开始','导航继续','继续上次导航','继续上一次导航'],
                         "navigation.cancel_navigation": ['结束导航','导航取消','退出导航','导航结束','撤销导航','取消导航','关闭导航','导航关闭','导航退出'],
                         "phone_call.make_a_phone_call": ['打电话给','拨打电话','拨号','呼叫'],
                         "OTHERS":['傻*', '讲个笑话','笑话','你好啊','怎么样','天气','收音机','调频','蓝牙电话']
                        }

        for key, value in ngram_pattern.items():
            for word in value:
                if word in text:
                    return key, word

        return None, None


    def keywords_match(self, text):
        text_seg = list(jieba.cut(text))

        for kw in self.keywords:
            if kw in text_seg:
                return "music.play", kw

        return None, None