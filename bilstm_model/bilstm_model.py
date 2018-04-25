from bilstm_model.run_multi_task_rnn import create_model, create_model_context, FLAGS
from bilstm_model.data_utils import initialize_vocab
import tensorflow as tf
import numpy as np

_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]

class BiLSTMModel(object):

    def __init__(self, data_dir, train_dir):
        FLAGS.data_dir = data_dir
        FLAGS.train_dir = train_dir
        print("Applying Parameters:")
        for k, v in FLAGS.__dict__['__flags'].items():
            print('%s: %s' % (k, str(v)))
        print("Preparing data in %s" % FLAGS.data_dir)

        vocab_path = data_dir + "/in_vocab_10000.txt"
        tag_vocab_path = data_dir + "/out_vocab_10000.txt"
        label_vocab_path = data_dir + "/label.txt"

        self.vocab, self.rev_vocab = initialize_vocab(vocab_path)
        self.tag_vocab, self.rev_tag_vocab = initialize_vocab(tag_vocab_path)
        self.label_vocab, self.rev_label_vocab = initialize_vocab(label_vocab_path)

        self.sess = tf.Session()

        _, self.model = create_model(self.sess, len(self.vocab), len(self.tag_vocab), len(self.label_vocab))
        self.model.batch_size = 1

    def data_prepare(self, input_text):
        encoder_inputs, decoder_inputs= [], []

        words = input_text.strip().split(" ")
        encoder_input = [self.vocab.get(w, 1) for w in words]
        encoder_pad = [0] * (FLAGS.max_sequence_length - len(encoder_input))
        encoder_inputs.append(list(encoder_input+encoder_pad))

        decoder_input = []
        decoder_inputs.append(decoder_input + [0] * (FLAGS.max_sequence_length - len(decoder_input)))

        sequence_lens = [len(input_text.strip().split(" "))]

        batch_encoder_inputs = []
        batch_decoder_inputs = []
        batch_weights = []
        batch_sequence_length = np.array(sequence_lens, dtype=np.int32)
        batch_labels = [np.array([0 for batch_idx in range(1)], dtype=np.int32)]

        for length_idx in range(FLAGS.max_sequence_length):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.int32)
            )

        for length_idx in range(FLAGS.max_sequence_length):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.int32)
            )
            batch_weight = np.ones(1, dtype=np.float32)
            for batch_idx in range(1):
                if decoder_inputs[batch_idx][length_idx] == 0:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels, batch_sequence_length

    def check_is_chinese(self, ch):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        return False

    def whitespece_segment(self, input_text):
        text_list = input_text.strip()

        seg_text = []
        for idx in range(len(text_list) - 1):
            if self.check_is_chinese(text_list[idx]) or self.check_is_chinese(text_list[idx + 1]):
                seg_text.append(text_list[idx])
                seg_text.append(" ")
            else:
                seg_text.append(text_list[idx])
        seg_text.append(text_list[-1])
        text = "".join(seg_text)

        return text


    def predict_intent(self, input_text, context):

        input_text = self.whitespece_segment(input_text)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels, batch_sequence_length = self.data_prepare(input_text)

        step_outputs = self.model.joint_step(self.sess,
                                             encoder_inputs=batch_encoder_inputs,
                                             tags=batch_decoder_inputs,
                                             tag_weights=batch_weights,
                                             labels=batch_labels,
                                             batch_sequence_length=batch_sequence_length,
                                             bucket_id=0,
                                             forward_only=True)

        _, _, _, class_logits = step_outputs
        hyp_label = np.argmax(class_logits[0], 0)
        return self.rev_label_vocab[hyp_label]

    def extract_slot(self, input_text, context):

        input_text = self.whitespece_segment(input_text)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels, batch_sequence_length = self.data_prepare(
            input_text)

        step_outputs = self.model.joint_step(self.sess,
                                             encoder_inputs=batch_encoder_inputs,
                                             tags=batch_decoder_inputs,
                                             tag_weights=batch_weights,
                                             labels=batch_labels,
                                             batch_sequence_length=batch_sequence_length,
                                             bucket_id=0,
                                             forward_only=True)

        _, _, tagging_logits, _ = step_outputs

        hyp_tag_list = [self.rev_tag_vocab[np.argmax(x)] for x in tagging_logits[:batch_sequence_length[0]]]

        return hyp_tag_list

    def extract_non_slot(self, input_text):

        input_text = self.whitespece_segment(input_text)
        hyp_tag_list = ["O" for _ in range(len(input_text.strip().split(" ")))]

        return hyp_tag_list

class BiLSTMModelContext(object):

    def __init__(self, data_dir, train_dir):
        FLAGS.data_dir = data_dir
        FLAGS.train_dir = train_dir
        print("Applying Parameters:")
        for k, v in FLAGS.__dict__['__flags'].items():
            print('%s: %s' % (k, str(v)))
        print("Preparing data in %s" % FLAGS.data_dir)

        vocab_path = data_dir + "/in_vocab_10000.txt"
        tag_vocab_path = data_dir + "/out_vocab_10000.txt"
        label_vocab_path = data_dir + "/label.txt"
        context_vocab_path = data_dir + "/context_vocab_10000.txt"

        self.vocab, self.rev_vocab = initialize_vocab(vocab_path)
        self.tag_vocab, self.rev_tag_vocab = initialize_vocab(tag_vocab_path)
        self.label_vocab, self.rev_label_vocab = initialize_vocab(label_vocab_path)
        self.context_vocab, self.rev_context_vocab = initialize_vocab(context_vocab_path)

        self.sess = tf.Session()

        _, self.model = create_model_context(self.sess, len(self.vocab), len(self.tag_vocab), len(self.label_vocab), len(self.context_vocab))
        self.model.batch_size = 1

    def data_prepare(self, input_text, context_text):
        encoder_inputs, decoder_inputs, context_inputs= [], [], []

        words = input_text.strip().split(" ")
        encoder_input = [self.vocab.get(w, 1) for w in words]
        encoder_pad = [0] * (FLAGS.max_sequence_length - len(encoder_input))
        encoder_inputs.append(list(encoder_input+encoder_pad))

        decoder_input = []
        decoder_inputs.append(decoder_input + [0] * (FLAGS.max_sequence_length - len(decoder_input)))

        words = context_text.strip().split(" ")
        context_input = [self.context_vocab.get(w, 1) for w in words]
        context_pad = [0] * (FLAGS.max_sequence_length - len(context_input))
        context_inputs.append(list(context_pad+context_input))

        sequence_lens = [len(input_text.strip().split(" "))]
        context_sequence_lens = [len(context_text.strip().split(" "))]

        batch_encoder_inputs = []
        batch_decoder_inputs = []
        batch_context_inputs = []
        batch_weights = []
        batch_sequence_length = np.array(sequence_lens, dtype=np.int32)
        batch_context_sequence_length = np.array(context_sequence_lens, dtype=np.int32)
        batch_labels = [np.array([0 for batch_idx in range(1)], dtype=np.int32)]

        for length_idx in range(FLAGS.max_sequence_length):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.int32)
            )

        for length_idx in range(FLAGS.max_sequence_length):
            batch_context_inputs.append(
                np.array([context_inputs[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.int32)
            )

        for length_idx in range(FLAGS.max_sequence_length):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(1)], dtype=np.int32)
            )
            batch_weight = np.ones(1, dtype=np.float32)
            for batch_idx in range(1):
                if decoder_inputs[batch_idx][length_idx] == 0:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels, batch_sequence_length, batch_context_inputs, batch_context_sequence_length

    def check_is_chinese(self, ch):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        return False

    def whitespece_segment(self, input_text):
        text_list = input_text.strip()

        seg_text = []
        for idx in range(len(text_list) - 1):
            if self.check_is_chinese(text_list[idx]) or self.check_is_chinese(text_list[idx + 1]):
                seg_text.append(text_list[idx])
                seg_text.append(" ")
            else:
                seg_text.append(text_list[idx])
        seg_text.append(text_list[-1])
        text = "".join(seg_text)

        return text


    def predict_intent(self, input_text, context_text):

        input_text = self.whitespece_segment(input_text)
        context_text = self.whitespece_segment(context_text)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels, batch_sequence_length, batch_context_inputs, batch_context_sequence_length = self.data_prepare(input_text, context_text)

        step_outputs = self.model.joint_step(self.sess,
                                             encoder_inputs=batch_encoder_inputs,
                                             context_inputs=batch_context_inputs,
                                             tags=batch_decoder_inputs,
                                             tag_weights=batch_weights,
                                             labels=batch_labels,
                                             batch_sequence_length=batch_sequence_length,
                                             bucket_id=0,
                                             forward_only=True)

        _, _, _, class_logits = step_outputs
        hyp_label = np.argmax(class_logits[0], 0)
        return self.rev_label_vocab[hyp_label]

    def extract_slot(self, input_text, context_text):

        input_text = self.whitespece_segment(input_text)
        context_text = self.whitespece_segment(context_text)

        batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_labels, batch_sequence_length, batch_context_inputs, batch_context_sequence_length = self.data_prepare(
            input_text, context_text)

        step_outputs = self.model.joint_step(self.sess,
                                             encoder_inputs=batch_encoder_inputs,
                                             context_inputs=batch_context_inputs,
                                             tags=batch_decoder_inputs,
                                             tag_weights=batch_weights,
                                             labels=batch_labels,
                                             batch_sequence_length=batch_sequence_length,
                                             bucket_id=0,
                                             forward_only=True)

        _, _, tagging_logits, _ = step_outputs

        hyp_tag_list = [self.rev_tag_vocab[np.argmax(x)] for x in tagging_logits[:batch_sequence_length[0]]]

        return hyp_tag_list

    def extract_non_slot(self, input_text):

        input_text = self.whitespece_segment(input_text)
        hyp_tag_list = ["O" for _ in range(len(input_text.strip().split(" ")))]

        return hyp_tag_list