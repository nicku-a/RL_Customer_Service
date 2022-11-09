import torch
from transformers import T5Tokenizer
from E2E_TOD.modelling.T5Model import T5Gen_Model
from E2E_TOD.ontology import * # sos_eos_tokens, requestable_slots, all_reqslot, informable_slots, all_infslot

import json
 
class NLU():
        def __init__(self):
                model_path = './checkpoints/small/'
                self.tokenizer = T5Tokenizer.from_pretrained(model_path)

                # Opening JSON file
                with open('reverse_token_map.json') as json_file:
                        self.mapping = json.load(json_file)

                with open('synonyms.json') as json_file:
                        self.synonyms = json.load(json_file)

                special_tokens = sos_eos_tokens
                self.model = T5Gen_Model(model_path, self.tokenizer, special_tokens, dropout=0.0, 
                        add_special_decoder_token=True, is_training=False)
                self.model.eval()

                # prepare some pre-defined tokens and task-specific prompts
                self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
                self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]
                self.pad_token_id, self.sos_b_token_id, self.eos_b_token_id, self.sos_a_token_id, self.eos_a_token_id, \
                self.sos_r_token_id, self.eos_r_token_id, self.sos_ic_token_id, self.eos_ic_token_id = \
                self.tokenizer.convert_tokens_to_ids(['<_PAD_>', '<sos_b>', 
                '<eos_b>', '<sos_a>', '<eos_a>', '<sos_r>','<eos_r>', '<sos_d>', '<eos_d>'])
                self.bs_prefix_text = 'translate dialogue to belief state:'
                self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.bs_prefix_text))
                self.da_prefix_text = 'translate dialogue to dialogue action:'
                self.da_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.da_prefix_text))
                self.nlg_prefix_text = 'translate dialogue to system response:'
                self.nlg_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.nlg_prefix_text))
                self.ic_prefix_text = 'translate dialogue to user intent:'
                self.ic_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.ic_prefix_text))

        # def check_auto(self, dialogue_context):
        #         if dialogue_context in 
                # return True, {}
        
        def nl_to_diaact(self, dialogue_context):


                # passed, dialogue = self.check_auto(dialogue_context)


                # an example dialogue context
                # dialogue_context = "<sos_u> can i reserve a five star place for thursday night at 3:30 for 2 people <eos_u> <sos_r> i'm happy to assist you! what city are you dining in? <eos_r> <sos_u> seattle please. <eos_u>"
                # dialogue_context = "can i reserve a five star place for thursday night at 3:30 for 2 people?"
                # dialogue_context = "2 tickets please"
                # dialogue_context = "I want to watch hail caesar in seattle. Where is it showing and when does it start?"
                context_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(dialogue_context))

                # predict belief state 
                input_id = self.bs_prefix_id + [self.sos_context_token_id] + context_id + [self.eos_context_token_id]
                input_id = torch.LongTensor(input_id).view(1, -1)
                x = self.model.model.generate(input_ids = input_id, decoder_start_token_id = self.sos_b_token_id,
                        pad_token_id = self.pad_token_id, eos_token_id = self.eos_b_token_id, max_length = 128)
                x_string, x_list = self.model.tokenized_decode_list(x[0])

                # print('NLP OUTPUT: {}'.format(x_string))

                clean_slots = []
                entity_idxs = []
                clean_entities = []

                if len(x_list) > 0:
                        category = x_list[0][1:-1]
                        if 'movie' in category:
                                x_list = x_list[1:]

                                for i in range(len(x_list)):
                                        not_slotted = True
                                        for j in range(len(x_list)-i):
                                                slot = ' '.join(x_list[i:-j])
                                                if slot in self.mapping and not_slotted:
                                                        index_beg = (i, len(x_list)-j)
                                                        entity_idxs.append(i)
                                                        entity_idxs.append(len(x_list)-j)
                                                        not_slotted = False
                                                        clean_slots.append(self.mapping[slot])
                                
                                if len(entity_idxs) > 0:
                                        entity_idxs = entity_idxs[1:]

                                        for i, idx in enumerate(entity_idxs[:-1:2]):
                                                clean_entities.append(' '.join(x_list[idx:entity_idxs[i*2+1]]))
                                        clean_entities.append(' '.join(x_list[entity_idxs[-1]:]))

                # predict dialogue act
                input_id = self.da_prefix_id + [self.sos_context_token_id] + context_id + [self.eos_context_token_id]
                input_id = torch.LongTensor(input_id).view(1, -1)
                x = self.model.model.generate(input_ids = input_id, decoder_start_token_id = self.sos_a_token_id,
                        pad_token_id = self.pad_token_id, eos_token_id = self.eos_a_token_id, max_length = 128)
                x_string, x_list = self.model.tokenized_decode_list(x[0])
                # print('NLP 2 OUTPUT: {}'.format(x_string))

                clean_req_slots = []

                if len(x_list) > 0:
                        # category = x_list[0][1:-1]
                        x_list = x_list[1:]

                        # categories = [(x[1:-1], i) for i, x in enumerate(x_list) if '[' in x]                        
                        # for category, index in categories:
                        #         print(category)

                        adding = True
                        new_x_list = []
                        for i, x in enumerate(x_list):
                                if x in '[request]':
                                        adding = False
                                elif '[' in x:
                                        adding = True
                                elif adding == True:
                                        new_x_list.append(x)
                        
                        x_list = new_x_list

                if len(x_list) > 0:               
                        for i in range(len(x_list)):
                                not_slotted = True
                                clean_req_slot = ''
                                for j in range(len(x_list)-i):
                                        slot = ' '.join(x_list[i:-j])
                                        if slot in self.mapping and not_slotted:
                                                not_slotted = False
                                                clean_req_slot = self.mapping[slot]
                                                if clean_req_slot not in clean_req_slots:
                                                        clean_req_slots.append(clean_req_slot)
                                                break
                        
                        if len(x_list)==1 and x_list[0] in self.mapping and x_list[0] not in clean_req_slots:
                                clean_req_slots.append(x_list[0])

                        clean_req_slots = [clean_req_slots[0]] + [slot for i, slot in enumerate(clean_req_slots[1:]) if slot not in clean_req_slots[i]]

                # Dialogue format intent/inform slots/request slots
                # Example dialogue: request/moviename: room, date: friday/starttime, city, theater
                intent = "inform"
                inform_slots = ""
                request_slots = ""
                for slot, entity in zip(clean_slots, clean_entities):
                        if entity != "":
                                inform_slots = inform_slots + ", " + slot.lower() + ": " + entity.lower()

                if len(clean_req_slots) > 0:
                        for slot in clean_req_slots:
                                request_slots = request_slots + ", " + slot.lower()
                        intent = "request"

                dialogue = intent + "/" + inform_slots[2:] + "/" + request_slots[2:]

                return dialogue


if __name__ == "__main__":
        nlu = NLU()
        
        # dialogue = "I want to watch avengers in new york. where is it showing and when does it start?"
        
        dialogue = "That works. where in seattle is it showing?"
        
        print(nlu.nl_to_diaact(dialogue))