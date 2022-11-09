import torch
from deepinsight.extractor import patching_block_with_type_hook
import transformers

patching_block_with_type_hook(
    specify_block_types=[
        transformers.models.bert.modeling_bert.BertLayer,
        transformers.models.bert.modeling_bert.BertEmbeddings,
        transformers.models.bert.modeling_bert.BertSelfAttention,
        transformers.models.bert.modeling_bert.BertSelfOutput,
        transformers.models.bert.modeling_bert.BertAttention,
        transformers.models.bert.modeling_bert.BertIntermediate,
        transformers.models.bert.modeling_bert.BertOutput,
        transformers.models.bert.modeling_bert.BertEncoder,
        transformers.models.bert.modeling_bert.BertPooler,
        transformers.models.gpt2.modeling_gpt2.GPT2Attention,
        transformers.models.gpt2.modeling_gpt2.GPT2MLP,
        transformers.models.gpt2.modeling_gpt2.GPT2Block,
        transformers.models.gpt2.modeling_gpt2.GPT2PreTrainedModel,
        transformers.models.gpt2.modeling_gpt2.GPT2Model,
        transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel,
        transformers.models.gpt2.modeling_gpt2.GPT2ForSequenceClassification,
        transformers.models.t5.modeling_t5.T5DenseReluDense,
        transformers.models.t5.modeling_t5.T5DenseGatedGeluDense,
        transformers.models.t5.modeling_t5.T5LayerFF,
        transformers.models.t5.modeling_t5.T5Attention,
        transformers.models.t5.modeling_t5.T5LayerSelfAttention,
        transformers.models.t5.modeling_t5.T5LayerCrossAttention,
        transformers.models.t5.modeling_t5.T5Block,
        transformers.models.t5.modeling_t5.T5PreTrainedModel,
        transformers.models.t5.modeling_t5.T5Stack,
        transformers.models.t5.modeling_t5.T5Model,
        transformers.models.t5.modeling_t5.T5ForConditionalGeneration,
        transformers.models.t5.modeling_t5.T5EncoderModel,
    ],
    export_dir="/tmp/hg_bert_block",
)

device = "cuda:0"


def bert_run():
    from transformers import BertConfig, AutoModelForMaskedLM
    from transformers.models.bert import BertForMaskedLM, BertLayer

    # from datasets import load_dataset

    torch.manual_seed(42)
    config = BertConfig()
    model = AutoModelForMaskedLM.from_config(config).cuda()
    # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    input_ids = torch.randint(0, config.vocab_size, (4, 512)).cuda()
    decoder_ids = torch.randint(0, config.vocab_size, (4, 512)).cuda()
    eval_context = torch.randint(0, config.vocab_size, (1, 512)).cuda()

    model.train()
    train_inputs = {"input_ids": input_ids, "labels": decoder_ids}
    outputs = model(**train_inputs)


def gpt2_run():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_config(config).to(device)
    # optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    input_ids = torch.randint(0, config.vocab_size, (4, 512)).to(device)
    decoder_ids = torch.randint(0, config.vocab_size, (4, 512)).to(device)

    eval_context = torch.randint(0, config.vocab_size, (1, 1024)).to(device)

    train_inputs = {"input_ids": input_ids, "labels": decoder_ids}
    eval_inputs = {
        "input_ids": eval_context,
    }

    outputs = model(**train_inputs)

    # loss = outputs.loss
    # loss.backward()
    # self.optimizer.step()


def t5_run():
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    torch.manual_seed(1337)
    config = AutoConfig.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_config(config).to(device)
    # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    train_bs = 4
    eval_bs = 1
    input_ids = torch.randint(0, config.vocab_size, (train_bs, 512)).to(device)
    decoder_ids = torch.randint(0, config.vocab_size, (train_bs, 512)).to(device)

    eval_context = torch.randint(0, config.vocab_size, (eval_bs, 1024)).to(device)

    train_inputs = {"input_ids": input_ids, "labels": decoder_ids}
    # self.eval_inputs = {'input_ids': eval_context, 'decoder_input_ids': eval_context}

    outputs = model(**train_inputs)
    # loss = outputs.loss
    # loss.backward()


# bert_run()
# gpt2_run()
bert_run()
