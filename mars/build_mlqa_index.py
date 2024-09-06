import json

# Requires the MLQA dataset from: https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip (https://github.com/facebookresearch/MLQA)

# Saves the artice title for each passage, this information is not easily available in the dataset as published on Hugginface
langs = ["en", "de"]
for lang in langs:
    with open(f"multilingual_data/MLQA_V1/test/test-context-{lang}-question-{lang}.json", "r") as f:
        data = json.load(f)

    qa_article_map = {}

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qa_article_map[qa["id"]] = article["title"]

    with open(f"multilingual_data/mlqa_index_{lang}_test.json", "w") as f:
        json.dump(qa_article_map, f)
