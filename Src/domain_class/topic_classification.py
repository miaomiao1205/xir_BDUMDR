import json
import fasttext
import os

DBPEDIA_LABELS = {
    1: "Company", 2: "EducationalInstitution", 3: "Artist", 4: "Athlete",
    5: "OfficeHolder", 6: "MeanOfTransportation", 7: "Building", 8: "NaturalPlace",
    9: "Village", 10: "Animal", 11: "Plant", 12: "Album", 13: "Film", 14: "WrittenWork"
}

YAHOO_ANSWERS_LABELS = {
    1: "Society & Culture", 2: "Science & Mathematics", 3: "Health", 4: "Education & Reference",
    5: "Computers & Internet", 6: "Sports", 7: "Business & Finance", 8: "Entertainment & Music",
    9: "Family & Relationships", 10: "Politics & Government"
}

PREDEFINED_TOPICS = {
    "Books & Literature": ["WrittenWork"],
    "Science & Mathematics": ["Science & Mathematics", "EducationalInstitution"],
    "Life & Health": ["Health", "Animal", "Plant"],
    "Jobs & Education": ["Education & Reference"],
    "Computers & Internet": ["Computers & Internet", "Company"],
    "Sports": ["Sports", "Athlete"],
    "Business & Finance": ["Business & Finance", "OfficeHolder"],
    "Politics & Government": ["Politics & Government"],
    "Traffic & Transportation": ["MeanOfTransportation"],
    "Arts & Entertainment": ["Entertainment & Music", "Artist", "Album", "Film"],
    "Geography": ["Building", "Village", "NaturalPlace"],
    "Others": []
}

def map_to_predefined_topic(db_label, yahoo_label):
    for topic, keywords in PREDEFINED_TOPICS.items():
        if yahoo_label in keywords:
            return topic
    for topic, keywords in PREDEFINED_TOPICS.items():
        if db_label in keywords:
            return topic
    return "Others (OT)"


def classify_with_models(text, db_model, yahoo_model):
    db_pred = db_model.predict(text)
    db_label_id = int(db_pred[0][0].split("__label__")[1])
    db_label = DBPEDIA_LABELS.get(db_label_id, "Unknown")
    db_score = float(db_pred[1][0])

    yahoo_pred = yahoo_model.predict(text)
    yahoo_label_id = int(yahoo_pred[0][0].split("__label__")[1])
    yahoo_label = YAHOO_ANSWERS_LABELS.get(yahoo_label_id, "Unknown")
    yahoo_score = float(yahoo_pred[1][0])

    return db_label, yahoo_label, db_score, yahoo_score

def process_single_file(input_file, output_file, db_model, yahoo_model):
    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            data = json.loads(line.strip())
            passages = data.get("positive_passages", [])

            if not passages:
                data["topic"] = "Others (OT)"
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            doc_text = passages[0].get("text", "").replace('"', "").replace('\n', " ")
            if not doc_text:
                data["topic"] = "Others (OT)"
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                continue

            db_label, yahoo_label, db_score, yahoo_score = classify_with_models(
                doc_text, db_model, yahoo_model
            )
            topic = map_to_predefined_topic(db_label, yahoo_label)

            data["topic"] = topic
            outfile.write(json.dumps(data, ensure_ascii=False) + "\n")

def main():
    langs = ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko',
             'ru', 'sw', 'te', 'th', 'zh']

    input_dir = "./Data/train_aug"
    output_dir = "./Data/train_label"
    os.makedirs(output_dir, exist_ok=True)

    db_model = fasttext.load_model("./PLM/dbpedia.bin")
    yahoo_model = fasttext.load_model("./PLM/yahoo_answers.bin")

    for lang in langs:
        input_file = os.path.join(input_dir, f"{lang}.jsonl")
        output_file = os.path.join(output_dir, f"{lang}_with_topic.jsonl")
        print(f"Processing {lang}...")
        process_single_file(input_file, output_file, db_model, yahoo_model)

if __name__ == "__main__":
    main()
