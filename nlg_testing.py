from nlg import nlg

# ################################################################################
# # load trained NLU model
# ################################################################################
# nlu_model_path = './deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p'
# nlu_model = nlu()
# nlu_model.load_nlu_model(nlu_model_path)

# nat_lang = 'I want to watch hail caesar in seattle. Where is it showing and when does it start?'

# da = nlu_model.generate_dia_act(nat_lang)

# print(da)

class NLG():
    def __init__(self):
        ################################################################################
        # load trained NLG model
        ################################################################################
        self.nlg_model_path = './nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p'
        self.diaact_nl_pairs = './nlg/dia_act_nl_pairs.v6.json'
        self.nlg_model = nlg()
        self.nlg_model.load_nlg_model(self.nlg_model_path)
        self.nlg_model.load_predefine_act_nl_pairs(self.diaact_nl_pairs)


    def diaact_to_nl(self, dia_act):
        # dia_act = {'diaact': 'inform', 'inform_slots': {'date': 'tomorrow', 'city': 'seattle'}, 'request_slots': {}}
        # dia_act = {'diaact': 'request', 'inform_slots': {}, 'request_slots': {'date': 'UNK', 'numberofpeople': 'UNK'}}

        nl = str(self.nlg_model.convert_diaact_to_nl(dia_act, 'agt'))

        if nl[:2] == "b'":
            nl = nl[2:-1]
        if dia_act['diaact'] == 'request' and nl[-1] != '?':
            nl = nl + '?'

        return nl
        