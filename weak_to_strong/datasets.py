import functools
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Optional
import random
import time

import sys

current_directory = sys.path.pop(0)

from datasets import Dataset as HfDataset
from datasets import load_dataset as hf_load_dataset

sys.path.insert(0, current_directory)

@dataclass
class DatasetConfig:
    # split -> unshuffled dataset of items
    loader: Callable[[str], HfDataset]
    # formats items to have keys 'txt' and 'hard_label', takes a random.Random rng
    formatter: Callable[[Any], Any]


# mapping from dataset name to load function and format function
_REGISTRY: dict[str, DatasetConfig] = {}


def register_dataset(name: str, config: DatasetConfig):
    _REGISTRY[name] = config


def load_dataset(ds_name: str, seed: int = 0, split_sizes: Optional[dict] = None):
    if split_sizes is None:
        split_sizes = dict(train=None, test=None)

    if ds_name not in _REGISTRY:
        raise ValueError(f"Unknown dataset {ds_name}, please register")
    cfg = _REGISTRY[ds_name] #cfg = config(loader, formatter)
    results = {}
    for split, n_docs in split_sizes.items():
        ds = cfg.loader(split)
        try:
            ds = ds.select(range(n_docs))
        except IndexError as e:
            print(f"Warning {ds_name} has less than {n_docs} docs, using all: {e}")
        #print(ds)
        ds_name_wo_rng = ['sciq_modify3', 'esnli_modify2', 'esnli_modify3', 'esnli_modify5', 'mbpp']
        if ds_name in ds_name_wo_rng:
            ds = ds.map(cfg.formatter)
        else:
            ds = ds.map(functools.partial(cfg.formatter, rng=Random(seed)))
        #ds = ds.map(lambda ex: {'hard_label': 3})
        #ds = ds.map(lambda ex: {'txt': 'hello world'})
        #print(type(ds))
        #print(ds['hard_label'][:100])
        #print(ds['txt'][0])
        if ds_name == 'sciq_modify3':
            # here, hardlabel already binarized
            ds = ds.map(
                lambda ex: {"soft_label": [1.0 if ((0-ex['hard_label'])==0) else 0.0, 1.0 if ((1-ex['hard_label'])==0) else 0.0,
                                           1.0 if ((2-ex['hard_label'])==0) else 0.0, 1.0 if ((3-ex['hard_label'])==0) else 0.0]
                            }
            )
        elif ds_name == 'esnli_modify2' or ds_name == 'esnli_modify3' or ds_name == 'esnli_modify5':
            ds = ds.map(
                lambda ex: {"soft_label": [1.0 if ((0 - ex['hard_label']) == 0) else 0.0,
                                           1.0 if ((1 - ex['hard_label']) == 0) else 0.0,
                                           1.0 if ((2 - ex['hard_label']) == 0) else 0.0,]
                            }
            )
        else:
            ds = ds.map(
                lambda ex: {"soft_label": [1 - float(ex["hard_label"]), float(ex["hard_label"])]}
            )
        ds = ds.shuffle(seed=seed)  # shuffling a bit pointless for test set but wtv
        results[split] = ds
    return results


def tokenize_dataset(
    raw_ds: HfDataset,
    tokenizer: Callable,
    max_ctx: int,
):
    """
    This function prepares the dataset for training. It takes the raw dataset, a formatting function,
    a tokenizer, a maximum context length

    Parameters:
    raw_ds: The raw dataset to be processed.load
    tokenizer: The tokenizer to be used on the formatted dataset.
    max_ctx: The maximum context length for the tokenizer.

    Returns:
    ds: The processed and shuffled dataset ready for training.
    """

    def process_function(res):
        toks = tokenizer(res["txt"])
        return dict(
            input_ids=toks["input_ids"],
        )

    ds = raw_ds.map(process_function, batched=False).filter(lambda x: len(x["input_ids"]) < max_ctx)
    return ds


def hf_loader(*hf_name, split_names=None):
    if split_names is None:
        split_names = dict()
    return lambda split: hf_load_dataset(*hf_name, split=split_names.get(split, split))


##########
# ACTUAL DATASETS
##########


def format_amazon_polarity(ex, rng):
    return dict(txt=f"{ex['title']} {ex['content']}", hard_label=ex["label"])


register_dataset(
    "amazon_polarity",
    DatasetConfig(loader=hf_loader("amazon_polarity"), formatter=format_amazon_polarity),
)


def format_sciq(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    txt = f"Q: {ex['question']} A: {ans}"
    return dict(txt=txt, hard_label=hard_label)



register_dataset(
    "sciq",
    DatasetConfig(loader=hf_loader("sciq"), formatter=format_sciq),
)


def format_sciq_modify3(ex):
    answers = [ex['correct_answer'], ex['distractor1'], ex['distractor2'], ex['distractor3']]
    # random.seed(rng_tmp.random())
    # choice_seq = random.sample(range(4), 4)
    hash_value = hash(ex['correct_answer'])
    random.seed(hash_value)
    choice_seq = random.sample(range(4),4)
    txt = f"Q: {ex['question']}\nChoices: \n0: {answers[choice_seq[0]]}\n1: {answers[choice_seq[1]]}\n2: {answers[choice_seq[2]]}\n3: {answers[choice_seq[3]]}\n"
    if choice_seq[0] == 0:
        hard_label = 0
    elif choice_seq[0] == 1:
        hard_label = 1
    elif choice_seq[0] == 2:
        hard_label = 2
    else:
        hard_label = 3
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "sciq_modify3",
    DatasetConfig(loader=hf_loader("sciq"), formatter=format_sciq_modify3),
)

def format_sciq_modify4(ex, rng):
    answers = [ex['correct_answer'], ex['distractor1'], ex['distractor2'], ex['distractor3']]
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    instruction = "Here are one question and four candidate choices, check if the choice in the answer is correct."
    hash_value = hash(ex['correct_answer'])
    random.seed(hash_value)
    choice_seq = random.sample(range(4), 4)
    txt = (f"Instruction: {instruction}\nQ: {ex['question']}\nChoices: \nA: {answers[choice_seq[0]]}\nB: {answers[choice_seq[1]]}\nC:"
           f" {answers[choice_seq[2]]}\nD: {answers[choice_seq[3]]}\nAnswer: {ans}")
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "sciq_modify4",
    DatasetConfig(loader=hf_loader("sciq"), formatter=format_sciq_modify4)
)

def format_sciq_modify5(ex, rng):
    #based on modify4, prompting disabled
    answers = [ex['correct_answer'], ex['distractor1'], ex['distractor2'], ex['distractor3']]
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    #instruction = "Here are one question and four candidate choices, check if the choice in the answer is correct."
    hash_value = hash(ex['correct_answer'])
    random.seed(hash_value)
    choice_seq = random.sample(range(4), 4)
    txt = (f"Question: {ex['question']}\nChoices: \nA: {answers[choice_seq[0]]}\nB: {answers[choice_seq[1]]}\nC:"
           f" {answers[choice_seq[2]]}\nD: {answers[choice_seq[3]]}\nAnswer: {ans}")
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "sciq_modify5",
    DatasetConfig(loader=hf_loader("sciq"), formatter=format_sciq_modify5)
)

def format_sciq_modify6(ex, rng):
    #modify6: for explanations + self-play
    answers = [ex['correct_answer'], ex['distractor1'], ex['distractor2'], ex['distractor3']]
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        ans = rng.choice([ex["distractor1"], ex["distractor2"], ex["distractor3"]])
    instruction = ("Given one question, 4 candidate choices, and a potential answer, check if the potential answer"
                   "is correct or not. When making your decision, if there is an 'explanation' section, also critically"
                   "analyze and take into consideration the 'explanation' section, which is the reasoning behind "
                   "someone else's choice on the same task you are facing.")
    hash_value = hash(ex['correct_answer'])
    random.seed(hash_value)
    choice_seq = random.sample(range(4), 4)
    txt = (f"Instruction: {instruction}\nQ: {ex['question']}\nChoices: \nA: {answers[choice_seq[0]]}\nB: {answers[choice_seq[1]]}\nC:"
           f" {answers[choice_seq[2]]}\nD: {answers[choice_seq[3]]}\nAnswer: {ans}")
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "sciq_modify6",
    DatasetConfig(loader=hf_loader("sciq"), formatter=format_sciq_modify6)
)

def format_anthropic_hh(ex, rng):
    hard_label = int(rng.random() < 0.5)
    txt = ex["chosen"] if hard_label else ex["rejected"]
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "anthropic_hh",
    DatasetConfig(loader=hf_loader("Anthropic/hh-rlhf"), formatter=format_anthropic_hh),
)

def format_anthropic_hh_modify1(ex, rng):
    hard_label = int(rng.random() < 0.5)
    instruction = ("Presented are 2 dialogues between human and assistant, decide whether the one chosen in answer should be chosen,"
                   "with the other dialogue rejected")
    order = int(rng.random() < 0.5)
    if hard_label:
        answer = ex['chosen']
    else:
        answer = ex['rejected']
    if order:
        txt = f"Instruction: {instruction}\nDialogue1: {ex['chosen']}\nDialogue2{ex['rejected']}\nAnswer: {answer}"
    else:
        txt = f"Instruction: {instruction}\nDialogue1: {ex['rejected']}\nDialogue2{ex['chosen']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "anthropic_hh_modify1",
    DatasetConfig(loader=hf_loader("Anthropic/hh-rlhf"), formatter=format_anthropic_hh_modify1),
)

def format_anthropic_hh_modify2(ex, rng):
    hard_label = int(rng.random() < 0.5)
    #instruction = ("Presented are 2 dialogues between human and assistant, decide whether the one chosen in answer should be chosen,"
    #               "with the other dialogue rejected")
    order = int(rng.random() < 0.5)
    if hard_label:
        answer = ex['chosen']
    else:
        answer = ex['rejected']
    if order:
        txt = f"Dialogue1: {ex['chosen']}\nDialogue2{ex['rejected']}\nAnswer: {answer}"
    else:
        txt = f"Dialogue1: {ex['rejected']}\nDialogue2{ex['chosen']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "anthropic_hh_modify2",
    DatasetConfig(loader=hf_loader("Anthropic/hh-rlhf"), formatter=format_anthropic_hh_modify2),
)

def format_cosmosqa(ex, rng):
    true_answer = ex["answer" + str(ex["label"])]
    if "None of the above choices ." in true_answer:
        hard_label = 0
    else:
        assert "None of the above choices" not in true_answer, true_answer
        hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer = true_answer
    else:
        candidate_answers = [ex["answer" + str(i)] for i in range(4)]
        answer = rng.choice([x for x in candidate_answers if x != true_answer])
    txt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "cosmos_qa",
    DatasetConfig(
        loader=hf_loader("cosmos_qa", split_names=dict(test="validation")),
        formatter=format_cosmosqa,
    ),
)


def format_boolq(ex, rng):
    hard_label = int(ex["answer"])
    txt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}"
    return dict(txt=txt, hard_label=hard_label)


register_dataset(
    "boolq",
    DatasetConfig(
        loader=hf_loader("boolq", split_names=dict(test="validation")), formatter=format_boolq
    ),
)


def format_socialiqa(ex, rng):
    if ex["label"] == 1:
        true_answer=ex["answerA"]
    elif ex["label"] == 2:
        true_answer=ex["answerB"]
    else:
        true_answer=ex["answerC"]
    ans=rng.choice([ex["answerA"], ex["answerB"], ex["answerC"]])
    if ans == true_answer:
        hard_label = 1
    else:
        hard_label = 0
    txt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {ans}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "social_i_qa",
    DatasetConfig(
        loader=hf_loader("social_i_qa", split_names=dict(test="validation")), formatter=format_socialiqa
    ),
)

def format_socialiqa_modify6(ex, rng):
    all_answers = [ex['answerA'],ex['answerB'],ex['answerC']]
    if ex["label"] == 1:
        true_answer=ex["answerA"]
        false_answers = [ex['answerB'], ex['answerC']]
    elif ex["label"] == 2:
        true_answer=ex["answerB"]
        false_answers = [ex['answerA'], ex['answerC']]
    else:
        true_answer=ex["answerC"]
        false_answers = [ex['answerB'], ex['answerA']]
    instruction = ("Here is a question presented together with context, 3 choices, and an answer provided that selects one of the 3 choices,"
                   "determine if the answer is correct")
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer = true_answer
    else:
        answer = rng.choice(false_answers)

    hash_value = hash(true_answer)
    random.seed(hash_value)
    choice_seq = random.sample(range(3), 3)

    txt = (f"Instruction: {instruction}\nContext: {ex['context']}\nQuestion: {ex['question']}\nChoices:\nA: "
           f"{all_answers[choice_seq[0]]}\nB: {all_answers[choice_seq[1]]}\nC: {all_answers[choice_seq[2]]}\n"
           f"Answer provided: {answer}")
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "social_i_qa_modify6",
    DatasetConfig(
        loader=hf_loader("social_i_qa", split_names=dict(test="validation")), formatter=format_socialiqa_modify6
    ),
)

def format_socialiqa_modify5(ex, rng):
    #prompting disabled
    all_answers = [ex['answerA'],ex['answerB'],ex['answerC']]
    if ex["label"] == 1:
        true_answer=ex["answerA"]
        false_answers = [ex['answerB'], ex['answerC']]
    elif ex["label"] == 2:
        true_answer=ex["answerB"]
        false_answers = [ex['answerA'], ex['answerC']]
    else:
        true_answer=ex["answerC"]
        false_answers = [ex['answerB'], ex['answerA']]
    #instruction = ("Here is a question presented together with context, 3 choices, and an answer provided that selects one of the 3 choices,"
    #               "determine if the answer is correct")
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer = true_answer
    else:
        answer = rng.choice(false_answers)

    hash_value = hash(true_answer)
    random.seed(hash_value)
    choice_seq = random.sample(range(3), 3)

    txt = (f"Context: {ex['context']}\nQuestion: {ex['question']}\nChoices:\nA: "
           f"{all_answers[choice_seq[0]]}\nB: {all_answers[choice_seq[1]]}\nC: {all_answers[choice_seq[2]]}\n"
           f"Answer provided: {answer}")
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "social_i_qa_modify5",
    DatasetConfig(
        loader=hf_loader("social_i_qa", split_names=dict(test="validation")), formatter=format_socialiqa_modify5
    ),
)

def format_super_glue(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        if int(ex['label'][0]) == 0:
            ans = "entailment"
        else:
            ans = "not entailment"
    else:
        if int(ex['label'][0]) == 0:
            ans = "not entailment"
        else:
            ans = "entailment"
    txt = f"Sentence1: {ex['sentence1']}\nSentence2: {ex['sentence2']}\nAnswer: {ans}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "super_glue",
    DatasetConfig(
        loader=hf_loader("super_glue"), formatter=format_super_glue
    ),
)

def format_quartz(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        answer=ex['answerKey']
    else:
        if ex['answerKey'] == 'A':
            answer = 'B'
        else:
            answer = 'A'
    choices = f"A. {ex['choices']['text'][0]}; B. {ex['choices']['text'][1]}"
    txt = f"Question: {ex['question']}\nChoices: {choices}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "quartz",
    DatasetConfig(
        loader=hf_loader("quartz", split_names=dict(test="validation")), formatter=format_quartz
    ),
)

def format_piqa(ex, rng):
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        if ex['label'] == 0:
            answer = ex['sol1']
        else:
            answer = ex['sol2']
    else:
        if ex['label'] == 0:
            answer = ex["sol2"]
        else:
            answer = ex["sol1"]
    txt = f"Goal: {ex['goal']}\nSolution: {answer}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "piqa",
    DatasetConfig(
        loader=hf_loader("piqa"), formatter=format_piqa
    ),
)

def format_quail(ex, rng):
    hard_label = int(rng.random() < 0.5)
    true_answer = ex['answers'][ex['correct_answer_id']]
    if hard_label:
        answer = true_answer
    else:
        candidate_answers = ex['answers']
        answer = rng.choice([x for x in candidate_answers if x != true_answer])
    txt = f"Context: {ex['context']}\nQuestion: {ex['question']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "quail",
    DatasetConfig(
        loader=hf_loader("quail", split_names=dict(test="validation")), formatter=format_quail
    ),
)

def format_esnli(ex, rng):
    hard_label = int(rng.random() < 0.5)
    esnli_labels={0:'entailment', 1:'neutral', 2:'contradiction'}
    all_labels = ['entailment', 'neutral', 'contradiction']
    true_label = esnli_labels[ex['label']]
    if hard_label:
        answer = true_label
    else:
        answer = rng.choice([x for x in all_labels if x != true_label])
    txt = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "esnli",
    DatasetConfig(
        loader=hf_loader("esnli", split_names=dict(test="validation")), formatter=format_esnli
    ),
)

def format_esnli_modify1(ex, rng):
    hard_label = int(rng.random() < 0.5)
    esnli_labels = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    all_labels = ['entailment', 'neutral', 'contradiction']
    true_label = esnli_labels[ex['label']]
    if hard_label:
        answer = true_label
    else:
        answer = rng.choice([x for x in all_labels if x != true_label])

    txt = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nAnswer: {answer}\nExplanation: {ex['explanation_1']}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "esnli_modify1",
    DatasetConfig(
        loader=hf_loader("esnli", split_names=dict(test="validation")), formatter=format_esnli_modify1
    ),
)

def format_esnli_modify2(ex):
    #esnli: 3 classes without explanations
    answers = ['true', 'neutral', 'false']
    hash_value = hash(ex['premise'])
    random.seed(hash_value)
    txt = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nChoice1: {answers[0]}\nChoice2: {answers[1]}\nChoice3: {answers[2]}"
    return dict(txt=txt, hard_label=ex['label'])

register_dataset(
    "esnli_modify2",
    DatasetConfig(
        loader=hf_loader("esnli", split_names=dict(test="validation")), formatter=format_esnli_modify2
    ),
)

def format_esnli_modify3(ex):
    # esnli: 3 classes with explanations
    answers = ['true', 'neutral', 'false']
    hash_value = hash(ex['premise'])
    random.seed(hash_value)
    txt = (f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nChoice1: {answers[0]}\nChoice2: {answers[1]}\n"
           f"Choice3: {answers[2]}\nExplanation: {ex['explanation_1']}")
    return dict(txt=txt, hard_label=ex['label'])

register_dataset(
    "esnli_modify3",
    DatasetConfig(
        loader=hf_loader("esnli", split_names=dict(test="validation")), formatter=format_esnli_modify3
    ),
)

def format_esnli_modify4(ex, rng):
    #corrected label names for original e-snli
    hard_label = int(rng.random() < 0.5)
    esnli_labels={0:'true', 1:'neutral', 2:'false'}
    all_labels = ['true', 'neutral', 'false']
    true_label = esnli_labels[ex['label']]
    if hard_label:
        answer = true_label
    else:
        answer = rng.choice([x for x in all_labels if x != true_label])
    txt = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nAnswer: {answer}"
    return dict(txt=txt, hard_label=hard_label)

register_dataset(
    "esnli_modify4",
    DatasetConfig(
        loader=hf_loader("esnli", split_names=dict(test="validation")), formatter=format_esnli_modify4
    ),
)

def format_esnli_modify5(ex):
    #3-class esnli with prompting, can be used for 0-shot
    prompt = ("Based on the premise, classify the hypothesis into 3 classes: [true, neutral, false]. Answer with one word: 'true', "
              "'neutral', or 'false'.")
    answers = ['true', 'neutral', 'false']
    hash_value = hash(ex['premise'])
    random.seed(hash_value)
    txt = (f"{prompt}\nPremise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\nChoice1: {answers[0]}\nChoice2: {answers[1]}\n"
           f"Choice3: {answers[2]}\nThe answer is: ")
    return dict(txt=txt, hard_label=ex['label'])

register_dataset(
    "esnli_modify5",
    DatasetConfig(
        loader=hf_loader("esnli", split_names=dict(test="validation")), formatter=format_esnli_modify5
    ),
)

def format_mbpp(ex):
    txt = f"{ex['text']}\n###\n{ex['code']}"
    return dict(txt=txt, hard_label=1)

register_dataset(
    "mbpp",
    DatasetConfig(
        loader=hf_loader("mbpp", split_names=dict(test='validation')), formatter=format_mbpp
    ),
)

# def format_esnli_modify1(ex, rng):
#     hard_label = int(rng.random() < 0.5)
#     esnli_labels={0:'true', 1:'maybe', 2:'false'}
#     all_labels = ['true', 'maybe', 'false']
#     true_label = esnli_labels[ex['label']]
#     if hard_label:
#         answer = true_label
#     else:
#         answer = rng.choice([x for x in all_labels if x != true_label])
#     txt = f"{ex['premise']}\nQuestion: {ex['hypothesis']}\n{esnli_labels[ex['label']]}\n"
#     return dict(txt=txt, hard_label=hard_label)
#
# register_dataset(
#     "esnli_modify1",
#     DatasetConfig(
#         loader=hf_loader("esnli", split_names=dict(test="validation")), formatter=format_esnli_modify1
#     ),
# )


VALID_DATASETS: list[str] = list(_REGISTRY.keys())

"""
from datasets import disable_caching
disable_caching()

from weak_to_strong.datasets import load_dataset, VALID_DATASETS
import numpy as np

ds_name = "boolq"
print(VALID_DATASETS)

ds = load_dataset(ds_name, split_sizes=dict(train=500, test=10))
train = list(ds['train'])
test = list(ds['test'])
print(test[0])
print(np.mean([x['hard_label'] for x in train]))
"""
