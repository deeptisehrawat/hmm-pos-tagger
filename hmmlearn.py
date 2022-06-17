import json
import sys

model_file_name = "hmmmodel.txt"


def calculate_counts(file_text):
    tags = dict()
    transition_prob = dict()
    emission_prob = dict()
    file_len = 0

    for sentence in file_text:
        file_len += 1
        prev_tag = "start"
        # tags[prev_tag] = tags.get(prev_tag, 0) + 1
        sequence = sentence.split(" ")
        for idx in range(len(sequence)+1):
            word = "end" if idx == len(sequence) else sequence[idx].rsplit("/", 1)[0]
            tag = "end" if idx == len(sequence) else sequence[idx].rsplit("/", 1)[1].rstrip()
            tags[tag] = tags.get(tag, 0) + 1

            # transition_prob
            if prev_tag in transition_prob:
                if tag in transition_prob[prev_tag]:
                    transition_prob[prev_tag][tag] = transition_prob[prev_tag][tag] + 1
                else:
                    transition_prob[prev_tag][tag] = 1
            else:
                # prev_tag_tag_dict = dict()
                # prev_tag_tag_dict[tag] = 1
                # transition_prob[prev_tag] = prev_tag_tag_dict
                transition_prob[prev_tag] = dict()
                transition_prob[prev_tag][tag] = 1
            prev_tag = tag

            # emission_prob
            if tag in emission_prob:
                if word in emission_prob[tag]:
                    emission_prob[tag][word] = emission_prob[tag][word] + 1
                else:
                    emission_prob[tag][word] = 1
            else:
                # tag_word_dict = dict()
                # tag_word_dict[word] = 1
                emission_prob[tag] = dict()
                emission_prob[tag][word] = 1

    tags["start"] = file_len
    tags["end"] = file_len

    return tags, transition_prob, emission_prob


def learn_model(training_fp):
    training_file_text = open(training_fp, 'r', encoding='utf8')
    tags, transition_prob, emission_prob = calculate_counts(training_file_text)

    # total_tag_count = sum(tags.values())
    for prev_tag in transition_prob:
        count = tags[prev_tag]
        for tag in transition_prob[prev_tag]:
            transition_prob[prev_tag][tag] = transition_prob[prev_tag][tag] / count

    for tag in emission_prob:
        count = tags[tag]
        for word in emission_prob[tag]:
            emission_prob[tag][word] = emission_prob[tag][word] / count

    transition_prob["end"] = dict()
    parameters = {
        "tags": tags,
        "transition_probabilities": transition_prob,
        "emission_probabilities": emission_prob,
    }
    model_fp = open(model_file_name, 'w')
    model_fp.write(json.dumps(parameters))
    model_fp.close()
    training_file_text.close()


if __name__ == '__main__':
    # learn_model(sys.argv[1])
    training_fp = "./hmm-training-data/it_isdt_train_tagged.txt"
    # training_fp = "./hmm-training-data/myinput.txt"
    learn_model(training_fp)
