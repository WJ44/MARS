import json

langs = ["en", "de"]
for lang in langs:
    data = json.load(open(f"multilingual_data/MLQA_V1/dev/dev-context-{lang}-question-{lang}.json"))

    qa_article_map = {}

    for article in data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                qa_article_map[qa["id"]] = article["title"]

    with open(f"multilingual_data/mlqa_index_{lang}.json", "w") as f:
        json.dump(qa_article_map, f)
