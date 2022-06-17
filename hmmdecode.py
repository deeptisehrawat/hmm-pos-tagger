import json
import sys
from random import randint

import numpy

model_file_name = "hmmmodel.txt"
output_file_name = "hmmoutput.txt"


def predict_tags(sentence, transition_prob, emission_prob, tags, tags_dict, total_tag_count):
    word_row = []
    sentence = sentence.rstrip()
    sequence_list = sentence.split(" ")

    # word_list[len(word_list)-1] = word_list[len(word_list)-1].rstrip()
    for sequence in sequence_list:
        word = sequence.rsplit("/", 1)[0]
        word_row.append(word)
    word_row.append("end")

    viterbi = numpy.zeros((len(tags), len(word_row)))
    back_pointers = numpy.zeros((len(tags), len(word_row)))
    most_probable_end_tag_idx = -1

    for word_idx, word in enumerate(word_row):
        max_end_word_prob = 0
        prev_most_probable_max = most_probable_end_tag_idx
        most_probable_end_tag_idx = -1

        for tag_idx, tag in enumerate(tags):
            if word_idx == 0:
                viterbi[tag_idx][word_idx] = transition_prob["start"].get(tag, 0) * emission_prob[tag].get(word, 1)
                if viterbi[tag_idx][word_idx] > max_end_word_prob:
                    most_probable_end_tag_idx = tag_idx
                    max_end_word_prob = viterbi[tag_idx][word_idx]
            else:
                max_prob = 0
                # most_probable_tag_idx = randint(0, len(tags)-1)
                most_probable_tag_idx = prev_most_probable_max
                if 0 != emission_prob[tag].get(word, 0):
                    for prev_tag_idx, prev_tag in enumerate(tags):

                        transition = transition_prob[prev_tag].get(tag, tags_dict[tag]/total_tag_count)
                        emission = emission_prob[tag].get(word, 0)
                        prev = viterbi[prev_tag_idx][word_idx-1]
                        # print('**********************************')
                        # print(tag + '|' + prev_tag)
                        # print('transition: ', transition)
                        # print(tag + '|' + word)
                        # print('emission: ', emission)
                        # print('prev: ', prev)
                        prob = prev * transition * emission
                        # print('prob ->d ', prob)
                        if prob > max_prob:
                            max_prob = prob
                            most_probable_tag_idx = prev_tag_idx

                viterbi[tag_idx][word_idx] = max_prob
                back_pointers[tag_idx][word_idx] = most_probable_tag_idx
                # word_idx == len(word_row) - 1 and
                if max_prob > max_end_word_prob:
                    most_probable_end_tag_idx = tag_idx
                    max_end_word_prob = max_prob
        # print(viterbi)

    most_probable_end_tag_idx = int(back_pointers[most_probable_end_tag_idx][len(word_row)-1])

    tagged_sentence = "\n"
    for word_idx in range(len(sequence_list) - 1, -1, -1):
        word = sequence_list[word_idx]
        # tag = tags[int(back_pointers[most_probable_end_tag_idx][word_idx])]
        tag = tags[most_probable_end_tag_idx]
        most_probable_end_tag_idx = int(back_pointers[most_probable_end_tag_idx][word_idx])
        tagged_sentence = " " + word + "/" + tag + tagged_sentence

    return tagged_sentence.lstrip()


def pos_tagger(test_fp):
    # load hmm model
    model = open(model_file_name, 'r')
    parameters = json.load(model)
    model.close()
    tags_dict = parameters["tags"]
    transition_prob = parameters["transition_probabilities"]
    emission_prob = parameters["emission_probabilities"]
    total_tag_count = sum(tags_dict.values())

    test_file_text = open(test_fp, 'r', encoding='utf8')
    output_file = open(output_file_name, 'w', encoding='utf8')

    for sentence in test_file_text:
        # create viterbi matrix
        # viterbi = dict()
        tags = []
        for tag in tags_dict:
            if tag != "start":
                tags.append(tag)

        tagged_sentence = predict_tags(sentence, transition_prob, emission_prob, tags, tags_dict, total_tag_count)
        output_file.write(tagged_sentence)

    output_file.close()


if __name__ == '__main__':
    # pos_tagger(sys.argv[1])
    test_fp = "./hmm-training-data/it_isdt_dev_tagged.txt"
    # test_fp = "./hmm-training-data/mytrain.txt"
    pos_tagger(test_fp)
