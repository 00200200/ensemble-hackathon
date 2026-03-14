JetBrains Challenge Starter Kit

Welcome to the starter kit for the JetBrains Challenge at the EnsembleAI Hackathon 2026! It will guide you from the moment of downloading data to running a baseline solution and making your first submission. Please feel free to fork the starter kit to use it as the starting point for your solutions.
Objective

The objective of the competition is to implement a context collection strategy that yields the most accurate code completions when provided to LLMs. Please read the challenge description for details.
Getting Started
Data Preparation

The starter kit expects that data files are stored in the data folder. stage is the competition stage, for the files already available in the starter kit stage = "start". language is the language split, kotlin or python.

The structure for data is as follows:

data
├── {language}-{stage}.jsonl # Competition data
└── repositories-{language}-{stage} # Folder with repositories
    └── {owner}__{repository}-{revision} # Repository revision used for collecting context
        └── repository contents

To prepare data for the starter kit:

    Download the data for the respective stage from the shared folder. Please note: unpacked data takes ~10GB on disk.
    Put the {language}-{stage}.jsonl file (datapoints) and the {language}-{stage}.zip archive (repos) to the data folder.
    Run ./prepare_data.sh practice python, possibly replacing practice with the stage and python with kotlin.

Running the baselines

The starter kit contains two baselines in baselines.py and an option to modify prefix/suffix of the completion file.

    Selecting a random Python file from the repository.
    Selecting a single Python file according to the BM-25 metric.

To run the baselines:

    poetry install --no-root – install dependencies via poetry
    poetry run python baselines.py --stage practice --strategy random --lang python – run the baselines
        You can replace practice with public to generate a complete submission for the competition
        You can replace random with another strategy, e.g., bm25
        You can replace python with kotlin for another split
        You can provide --trim-prefix and/or trim-suffix to modify the used prefix and suffix of the completion file by trimming it to 10 lines
    The prediction file will be saved in the predictions folder.

Implementing your own strategy

Please look at the implementation of the baselines in baselines.py for an example. If the selected context contains multiple files, their parts included in the context should be separated by <|file_sep|>.
Prediction file format

The predictions are expected in a JSON Lines file, with each object having a context field:

{"context": "**context for prediction 1**"}
{"context": "**context for prediction 2**"}
...
{"context": "**context for prediction N**"}

The number and order of the objects should correspond to the objects in the input .jsonl file.

By default, the entire prefix and suffix of the file will be provided to the model. Optionally, you can submit your own version of the prefix and suffix in the completion file.
In that case, the format of an entry in the file is:

{"context": "context for prediction", "prefix": "custom prefix", "suffix": "custom suffix"}

Both prefix and suffix are optional in each entry: it is acceptable if each of the fields is only specified in some of the entries.
Submitting your solution

Go to the competition page, enter the competition, select the stage, and upload the generated prediction file.
Evaluation

The contexts are used to generate completions with three different models based on their similarity to the reference text with the ChrF score metric. The final score is the average across the scores of the three models. Please read the Evaluation section of the competition page for details.