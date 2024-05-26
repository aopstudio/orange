import re
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          BartForConditionalGeneration, BartTokenizer,)

ORANGE_RE_GENERATOR = 'relation_generator'
ORANGE_ENT_GENERATOR = 'entity_generator'

RELATIONS = [
    "Causes",
    "HasProperty",
    "MadeUpOf",
    "isAfter",
    "isBefore",
    "xReact",
    "xWant",
    "xReason",
    "xAttr",
    "Desires",
]

def formalize_tA(tA):
    tA = tA.strip()
    if tA.endswith('.'):
        tA = tA[:-1].strip() + '.'
    else:
        tA += '.'
    tA = tA.replace(' ,', ',')
    tA = tA.replace(" '", "'")
    return tA

def filter_words(words_prob):
    word_count = {}
    token1_count = {}
    word2_count = {}
    ret = []
    for words, prob, *_ in words_prob:
        filter_this = False

        token_count = {}
        for word in words:
            for token in word.split(' '):
                if token in token_count:
                    filter_this = True
                token_count[token] = 1
        if filter_this:
            prob *= 0.5


        if len(words) == 3 and words[0] == words[2]:
            continue
        token1 = words[0].split(' ')[0]
        if token1 not in token1_count:
            token1_count[token1] = 1
        else:
            token1_count[token1] += 1
            prob /= token1_count[token1]


        for word in words:
            if word not in word_count:
                word_count[word] = 0
            word_count[word] += 1
            prob /= word_count[word]

        if len(words) == 3:
            if words[2] not in word2_count:
                word2_count[words[2]] = 0
            word2_count[words[2]] += 1
            prob /= word2_count[words[2]]

        ret.append([words, prob])
    return sorted(ret, key=lambda x: x[1], reverse=True)

def post_process_template(tB):
    if tB.endswith('.') == False:
        tB += '.'
    return tB

from copy import deepcopy

def convert_for_print(arr):
    ret = deepcopy(arr)
    for i in range(len(ret)):
        ret[i][1] = round(ret[i][1], 7)
        if len(ret[i]) == 3:
            for j in range(len(ret[i][2])):
                ret[i][2][j] = round(ret[i][2][j], 7)
    return ret

def construct_template(words, templateA, if_then=False):
    if len(words) == 3:
        templates = [
            '{} <mask> {}.'.format(words[0], words[2]),
        ]
    elif len(words) == 1:
        templates = [
            '{} <mask>.'.format(words[0])]

    elif len(words) == 0:
        templates = []

    return templates

class BartInductor(object):
    def __init__(
        self, 
    ):
        self.orange_entity_generator_path = ORANGE_ENT_GENERATOR
        self.orange_relation_generator_path = ORANGE_RE_GENERATOR
        
        self.orange_entity_generator = BartForConditionalGeneration.from_pretrained(self.orange_entity_generator_path).cuda().eval().half()
        self.orange_relation_generator = BartForConditionalGeneration.from_pretrained(self.orange_relation_generator_path).cuda().eval().half()
        self.tokenizer = BartTokenizer.from_pretrained(ORANGE_ENT_GENERATOR)
        self.word_length = 2

        self.stop_sub_list = ['he', 'she', 'this', 'that', 'and', 'it', 'which', 'who', 'whose', 'there', 'they', '.', 'its', 'one',
                                'i', ',', 'the', 'nobody', 'his', 'her', 'also', 'only', 'currently', 'here', '()', 'what', 'where',
                                'why', 'a', 'some', '"', ')', '(', 'now', 'everyone', 'everybody', 'their', 'often', 'usually', 'you',
                                '-', '?', ';', 'in', 'on', 'each', 'both', 'him', 'typically', 'mostly', 'sometimes', 'normally',
                                'always', 'usually', 'still', 'today', 'was', 'were', 'but', 'although', 'current', 'all', 'have',
                                'has', 'later', 'with', 'most', 'nowadays', 'then', 'every', 'when', 'someone', 'anyone', 'somebody',
                                'anybody', 'any', 'being', 'get', 'getting', 'thus', 'under', 'even', 'for', 'can', 'rarely', 'never',
                                'may', 'generally', 'other', 'another', 'too', 'first', 'second', 'third', 'mainly', 'primarily',
                                'having', 'have', 'has']

        self.stop_size = len(self.stop_sub_list)
        for i in range(self.stop_size):
            if self.stop_sub_list[i][0].isalpha():
                temp = self.stop_sub_list[i][0].upper() + self.stop_sub_list[i][1:]
                self.stop_sub_list.append(temp)

        self.bad_words_ids = [self.tokenizer.encode(bad_word)[1:-1] for bad_word in ['also', ' also',';','the']]
        stop_index = self.tokenizer(self.stop_sub_list, max_length=4, padding=True)
        stop_index = torch.tensor(stop_index['input_ids'])[:, 1]
        stop_weight = torch.zeros(1, self.tokenizer.vocab_size).cuda()
        stop_weight[0, stop_index] -= 100
        self.stop_weight = stop_weight[0, :]

    def clean(self, text):
        segments = text.split('<mask>')
        if len(segments) == 3 and segments[2].startswith('.'):
            return '<mask>'.join(segments[:2]) + '<mask>.'
        else:
            return text

    def generate(self, inputs, k=10, topk=10):
        with torch.no_grad():
            tB_probs = self.generate_rule(inputs, k)
            ret = [t[0].replace('<ent0>','<mask>').replace('<ent2>','<mask>') for t in tB_probs]

            new_ret = []
            for temp in ret:
                temp = self.clean(temp.strip())
                if len(new_ret) < topk and temp not in new_ret:
                    new_ret.append(temp)
                    
            if len(new_ret) == 0:
                return new_ret
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(new_ret)
            kmeans = KMeans(n_clusters=min(len(new_ret),4), random_state=42)
            kmeans.fit(X)
            labels = kmeans.predict(X)
            centers = kmeans.cluster_centers_
            closest_sentences = []
            for i in range(kmeans.n_clusters):
                indices = np.where(labels == i)[0]
                distances = np.linalg.norm(X[indices] - centers[i], axis=1)
                if len(distances) == 0:
                    return []
                closest_index = indices[np.argmin(distances)]
                closest_sentence = new_ret[closest_index]
                closest_sentences.append(closest_sentence)
            indices = []
            for clo_sent in closest_sentences:
                index = new_ret.index(clo_sent)
                indices.append(index)
            data = list(zip(closest_sentences, indices))
            sorted_data = sorted(data, key=lambda x: x[1])
            sorted_sentences = [x[0] for x in sorted_data]
            return sorted_sentences

    def explore_mask(self, tA, k, tokens, prob, required_token, probs):
        if required_token == 0:
            return [[tokens, prob, probs]]
        if required_token <= self.word_length:
            k = min(k, 2)
        ret = []
        generated_ids = self.tokenizer(tA, max_length=128, padding='longest', return_tensors='pt')
        for key in generated_ids.keys():
            generated_ids[key] = generated_ids[key].cuda()
        mask_index = torch.where(generated_ids["input_ids"][0] == self.tokenizer.mask_token_id)
        generated_ret = self.orange_entity_generator(**generated_ids)
        logits = generated_ret[0]
        softmax = F.softmax(logits, dim=-1)
        mask_word = softmax[0, mask_index[0][0], :] + self.stop_weight
        top_k = torch.topk(mask_word, k, dim=0)
        for i in range(top_k[1].size(0)):
            token_s = top_k[1][i]
            prob_s = top_k[0][i].item()
            token_this = self.tokenizer.decode([token_s]).strip()
            if token_this[0].isalpha() == False or len(token_this) <= 2:
                continue
            index_s = tA.index(self.tokenizer.mask_token)
            tAs = tA[:index_s] + token_this + tA[index_s + len(self.tokenizer.mask_token):]
            tokens_this = [t for t in tokens]
            tokens_this.append(token_this)
            probs_new = deepcopy(probs)
            probs_new.append(prob_s)
            ret.extend(self.explore_mask(tAs, 1, tokens_this, prob_s * prob, required_token - 1,probs_new))
        return ret

    def extract_words_for_tA_bart(self, tA, k=10, print_it = False):
        spans = [t.lower().strip() for t in tA[:-1].split('<mask>')]

        generated_ids = self.tokenizer([tA],max_length=500, padding='longest', return_tensors='pt')['input_ids'].cuda()
        generated_ret = self.orange_entity_generator.generate(
            generated_ids,
            num_beams=10, 
            num_beam_groups = 5,
            diversity_penalty = 3.0,     
            num_return_sequences=10, 
            min_length=generated_ids.size(1),
            max_length=generated_ids.size(1)+20,
            output_scores=True,
            return_dict_in_generate=True
        )
        summary_ids = generated_ret['sequences']
        probs = F.softmax(generated_ret['sequences_scores'])
        txts = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in summary_ids]

        ret = []
        for i, txt in enumerate(txts):
            if tA.endswith('.'):
                if txt.endswith('.'):
                    txt = txt[:-1].strip()
                txt += '.'
            word_imcomplete = False
            prob = probs[i].item()
            words_i = []
            start_index = 0
            for j in range(len(spans)-1):
                span1 = spans[j]
                span2 = spans[j+1]
                if (span1 in txt.lower()[start_index:]) and (span2 in txt.lower()[start_index:]):
                    index1 = txt.lower().index(span1,start_index)+len(span1)
                    if span2 == '':
                        if txt[-1] == '.':
                            index2 = len(txt) -1
                        else:
                            index2 = len(txt)
                    else:
                        index2 = txt.lower().index(span2, start_index)

                    words_i.append(txt[index1:index2].strip())
                    start_index = index2
                else:
                    word_imcomplete = True
            if word_imcomplete:
                continue

            ret.append([words_i, prob])

        return sorted(ret, key=lambda x: x[1], reverse=True)[:k]


    def extract_words_for_tA(self, tA, k=10):
        word_mask_str = ' '.join([self.tokenizer.mask_token] * self.word_length)
        tA = tA.replace('<mask>', word_mask_str)
        mask_count = tA.count(self.tokenizer.mask_token)
        mask_probs = self.explore_mask(tA, k*20, [], 1.0, mask_count, [])
        ret = []
        visited_mask_txt = {}
        for mask, prob, probs in mask_probs:
            mask_txt = ' '.join(mask).lower()
            if mask_txt in visited_mask_txt:
                continue
            visited_mask_txt[mask_txt] = 1
            words = []
            probs_words = []
            for i in range(0,mask_count, self.word_length):
                words.append(' '.join(mask[i: i + self.word_length]))
                prob_word = 1.0
                for j in range(i, i + self.word_length):
                    prob_word *= probs[j]
                probs_words.append(prob_word)
            ret.append([words, prob, probs_words])
        return sorted(ret, key=lambda x: x[1], reverse=True)[:k]

    def extract_templateBs_batch(self, words_prob, tA, k, print_it = False):
        words_prob_sorted = []
        for (words, probA, *_) in words_prob:
            tokenized_word = self.tokenizer(words[0])
            words_prob_sorted.append([words,probA,len(tokenized_word['input_ids'])])
        words_prob_sorted.sort(key=lambda x:x[2])

        batch_size = 10
        templates = []
        index_words = {}
        ret = {}
        num_beams = 10
        for enum, (words, probA, *_) in enumerate(words_prob_sorted):
            template = construct_template(words, tA, False)
            templates.extend(template)
            for t in template:
                index_words[len(index_words)] = '\t'.join(words)
            if (len(templates) == batch_size) or enum==len(words_prob_sorted)-1:
                generated_ids = self.tokenizer(templates, padding="longest", return_tensors='pt')['input_ids'].cuda()
                generated_ret = self.orange_relation_generator.generate(generated_ids, num_beams=num_beams,
                                max_length=100,
                                num_return_sequences=num_beams,
                                bad_words_ids=self.bad_words_ids,
                                output_scores=True,
                                return_dict_in_generate=True,
                                )
                summary_ids = generated_ret['sequences'].reshape((len(templates),num_beams,-1))
                probs = F.softmax(generated_ret['sequences_scores'].reshape((len(templates),num_beams)),dim=1)
                for ii in range(summary_ids.size(0)):
                    txts = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                            summary_ids[ii]]
                    ii_template = []
                    words_ii = index_words[ii].split('\t')
                    for i, txt in enumerate(txts):
                        prob = probs[ii][i].item() * probA
                        txt = txt.lower()
                        txt = post_process_template(txt)
                        words_ii_matched = [word.lower() for word in words_ii] 
                        if words_ii_matched is None:
                            prob = 0.0
                        else:
                            for j, word in enumerate(words_ii_matched):
                                if (j == 0 or j==2) and word not in txt:
                                    prob = 0.0
                                else:
                                    if (j == 0 or j==2):    
                                        txt = txt.replace(word, '<ent{}>'.format(j), 1)
                        if txt.count(' ')+1<=3:
                            continue
                        ii_template.append([txt, prob])

                    for template, prob in ii_template:
                        if template.endswith('..'):
                            template = template[:-1]
                        if template not in ret:
                            ret[template] = 0.0
                        ret[template] += prob
                templates.clear()
                index_words.clear()
        return ret

    def generate_rule(self, tA, k=10, print_it = False):
        tA=formalize_tA(tA)
        if 'bart' in str(self.orange_entity_generator.__class__).lower():
            words_prob = self.extract_words_for_tA_bart(tA, k,print_it=print_it)
            words_prob = filter_words(words_prob)[:k]
        else:
            words_prob = self.extract_words_for_tA(tA, k)
            words_prob = filter_words(words_prob)[:k]

        tB_prob = self.extract_templateBs_batch(words_prob, tA, k,print_it=print_it)

        ret = []
        for k1 in tB_prob:
            ret.append([k1, tB_prob[k1]])
        ret = sorted(ret, key=lambda x: x[1], reverse=True)[:k]
        return ret