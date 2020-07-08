from src.models.spacy_embedding import spacy_embedding_workflow
from src.models.tfidf_ngram import tfidf_workflow
from src.utils.plot_categories import plot_workflow
from src.utils.args import *


def main():
    plot_workflow(
        pth=args.pth,
        category=args.category
    )
    tfidf_workflow(
        pth=args.pth,
        col_to_stem=args.col_to_stem,
        path_to_save_model=args.tfidf_model_pth
    )
    embedding_model = spacy_embedding_workflow(
        pth=args.pth,
        col_to_embed=args.col_to_embed,
        path_to_save_model=args.embedding_model_pth
    )


if __name__ == "__main__":
    main()