import argparse, pickle, re, os
from time import sleep

path_to_questions_mapping = 'path/to/questions/mapping.pkl'
path_to_filenames_path = 'path/to/filenames/path.txt'

def read_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.loads(f.read())
    return data

def write_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f, protocol=2)

def read_files(args):
    questions_mapping = read_pickle(path_to_questions_mapping)
    filenames_path = path_to_filenames_path

    with open(args.summaries_file, 'r', encoding='utf-8') as f:
        summaries = f.read().splitlines()

    with open(filenames_path, 'r', encoding='utf-8') as f:
        filenames = f.read().splitlines()

    return questions_mapping, summaries, filenames


def answer_questions(args):
    total_correct, total_questions = 0, 0

    questions_mapping, summaries, filenames = read_files(args)

    print('answer_questions')
    for i, (summary, art_hash) in enumerate(zip(summaries, filenames)):
        print("iter: "  str(i))
        entitized_summary = entitize(summary, questions_mapping[art_hash]['mapping'])
        curr_questions, curr_answers = zip(*[(q['question'], q['answer']) for q in questions_mapping[art_hash]['questions'].values()])
        num_questions = len(curr_questions)
        num_correct = 0

        if '@' in entitized_summary:
            acc = eval_acc(args.qa_id, [[entitized_summary]*num_questions, curr_questions, curr_answers, []])/100
            num_correct = acc * num_questions

        print("NUM CORRECT: " + str(num_correct))
        print("NUM QUESTIONS: " + str(num_questions))
        total_correct += num_correct
        total_questions += num_questions

    print("TOTAL CORRECT: " + str(total_correct))
    print("TOTAL QUESTIONS: " + str(total_questions))

def entitize(summary, entities):
    entitized_summary = summary
    for ent_id, ent_name in sorted(entities.items(), key=lambda item: len(item[1]), reverse=True): 
        entitized_summary = re.sub(r'\b' + re.escape(ent_name) + r'\b', ent_id, entitized_summary, flags=re.IGNORECASE)

    return entitized_summary

def eval_acc(qa_id, data):
    query_path = './queries'+str(qa_id)+'.pkl'
    rewards_path = './rewards'+str(qa_id)+'.txt'

    write_pickle(query_path, data)
    while(not os.path.isfile(rewards_path)):
        sleep(0.1)

    rewards_file = open(rewards_path, 'r')
    acc = rewards_file.read()
    os.remove(rewards_path)
    try:
        acc = float(acc)
    except Exception:
        acc = 0.0

    return acc


def main():
    parser = argparse.ArgumentParser(description='Answering to questions')
    parser.add_argument('--summaries_file', required=True, type=str)
    parser.add_argument('--qa_id', required=True, type=int)
    args = parser.parse_args()

    answer_questions(args)


if __name__ == "__main__":
   main()