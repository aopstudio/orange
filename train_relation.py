from transformers import BartForConditionalGeneration, BartTokenizerFast
import pandas as pd
from datasets import Dataset,load_dataset

model = BartForConditionalGeneration.from_pretrained("bart-base")
tokenizer = BartTokenizerFast.from_pretrained("bart-base")

def tokenize_function(examples):
    result = tokenizer(examples["text"],padding='max_length',max_length=100, truncation=True)
    result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    labels= tokenizer(examples["label"],padding='max_length',max_length=100, truncation=True)['input_ids']
    result['labels'] = labels
    return result

train_data_file = f'relation_output/train.txt'
df_train=pd.read_table(train_data_file, sep='\t', header=0)


train_dataset = Dataset.from_pandas(df_train)

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])


import collections

from transformers import default_data_collator

wwm_probability = 0


def whole_word_masking_data_collator(features):
    for feature in features:
        if 'word_ids' in feature:
            word_ids = feature.pop("word_ids")
            # Create a map between words and corresponding token indices
            mapping = collections.defaultdict(list)
            current_word_index = -1
            current_word = None
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    if word_id != current_word:
                        current_word = word_id
                        current_word_index += 1
                    mapping[current_word_index].append(idx)

    return default_data_collator(features)


downsampled_dataset = tokenized_train_dataset.train_test_split(
    train_size=45000, test_size=5000, seed=42
)


from transformers import TrainingArguments

batch_size = 100
logging_steps = len(downsampled_dataset["train"]) // batch_size

training_args = TrainingArguments(
    save_strategy="epoch",
    output_dir="./relation_generator",
    overwrite_output_dir="./relation_generator",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    logging_steps=logging_steps,
    num_train_epochs=10,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=whole_word_masking_data_collator,
    tokenizer=tokenizer,
)

trainer.train()