import argparse


def parser():
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('--pth', default='./src/data/sampled_data_consolidated_cat.csv', type=str)
    PARSER.add_argument('--category', default='POLITICS', type=str, choices=['ENTERTAINMENT', 'POLITICS'])
    PARSER.add_argument('--col_to_stem', default='headline', type=str)
    PARSER.add_argument('--col_to_embed', default='headline', type=str)
    PARSER.add_argument('--tfidf_model_pth', default='./src/models/best_model_tfidf/', type=str)
    PARSER.add_argument('--embedding_model_pth', default='./src/models/best_model_embedding/', type=str)

    ARGS = PARSER.parse_args()
    return ARGS

args = parser()


if __name__ == "__main__":
    pass