# -*- coding:utf8 -*-
import jieba
import json
from data_util.utils import load_json


class RuleMatcher(object):

    def __init__(self,
                 jieba_dict_path,
                 dict_path,
                 command_json_path,
                 ngram_json_path,
                 song_list_path):
        self.keyword_dict = {}
        self.song_list = []
        self.command_dict = {}
        self.ngram_dict = {}

        self._load_jieba_dict(jieba_dict_path)
        self._load_keywords(dict_path)
        self._load_song_list(song_list_path)
        self._load_commands(command_json_path)
        self._load_ngrams(ngram_json_path)

    def _load_jieba_dict(self, jieba_dict_path):
        jieba.load_userdict(jieba_dict_path)

    def _load_keywords(self, dict_path):
        self.keyword_dict = load_json(dict_path)

    def _load_song_list(self, song_list_path):
        with open(song_list_path) as fin:
            for line in fin:
                song = line.strip()
                self.song_list.append(song)

    def _load_commands(self, command_json_path):
        self.command_dict = load_json(command_json_path)

    def _load_ngrams(self, ngram_json_path):
        self.ngram_dict = load_json(ngram_json_path)

    def match(self, text, last_intent=None):

        intent, pattern = self.sentence_match(text)

        if intent is None:
            intent, pattern = self.song_match(text)

        if intent is None:
            intent, pattern = self.keywords_match(text)

        if intent is None:
            intent, pattern = self.number_match(text)

        if intent is None:
            intent, pattern = self.cancel_match(text, last_intent)

        if intent is None:
            intent, pattern = self.ngram_match(text)

        return intent, pattern

    def cancel_match(self, text, last_intent=None):
        if text != '取消':
            return None, None
        elif last_intent is None:
            return 'OTHERS', '#CANCELMATCH'
        elif 'phone' in last_intent:
            return 'phone_call.cancel', '#CANCELMATCH'
        elif 'navigation' in last_intent:
            return 'navigation.cancel_navigation', '#CANCELMATCH'
        elif 'music' in last_intent:
            return 'music.pause', '#CANCELMATCH'
        else:
            return 'OTHERS', '#CANCELMATCH'

    def sentence_match(self, text):
        for key, value in self.command_dict.items():
            for sent in value:
                if sent == text:
                    return key, "#COMMANDMATCH"

        return None, None

    def ngram_match(self, text):
        for key, value in self.ngram_dict.items():
            for word in value:
                if word in text:
                    return key, "#NGRAMMATCH"

        return None, None

    def song_match(self, text):
        if text in self.song_list:
            return "music.play", "#SONGMATCH"
        return None, None

    def number_match(self, text):
        if text.isdigit():
            if len(text) == 11 and text[0] == "1":
                return "phone_call.make_a_phone_call", "#NUMBERMATCH"
            else:
                return "OTHERS", "#NUMBERMATCH"
        else:
            return None, None

    def keywords_match(self, text):
        text_seg = list(jieba.cut(text))

        for k, v in self.keyword_dict.items():
            for word in v:
                if word in text_seg:
                    return k, "#KEYWORDMATCH"

        return None, None
